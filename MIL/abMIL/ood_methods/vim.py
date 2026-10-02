"""
VIM (Virtual-logit Matching) OOD Detection
Reference: Wang et al., "ViM: Out-Of-Distribution with Virtual-logit Matching", CVPR 2022
Adapted from: https://github.com/remic-othr/OpenMIBOOD
"""

import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm
import sys


class VIMPostprocessor:
    """
    VIM OOD scorer for AB-MIL models.

    Classifier 結構：bag_embedding (in_dim)
                      → fc1 → ReLU → hidden (256)
                      → fc2 → logits (2)

    VIM 操作在 fc2 的輸入空間（hidden, dim=256），
    取 fc2.weight 與 fc2.bias 作為分類超平面。

    score = -vlogit * alpha + energy  (higher → more ID)
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.setup_flag = False

        self.w     = None   # (C, 256)   fc2.weight
        self.b     = None   # (C,)       fc2.bias
        self.u     = None   # (256,)     principal anchor
        self.NS    = None   # (256, 256-dim)  null-space projection
        self.alpha = None   # float

        self._feature_id_train = None
        self._logit_id_train   = None
        self._eig_vals         = None
        self._eigen_vectors    = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_hidden(classifier, abmil, chief_data):
        """
        取得 fc2 的輸入 feature，即 relu(fc1(bag_embedding))。
        shape: (B, 256)
        """
        bag_embedding, _ = abmil(chief_data)            # (B, in_dim)
        hidden = F.relu(classifier.fc1(bag_embedding))  # (B, 256)
        return hidden

    @staticmethod
    def _unpack_batch(batch, device: str):
        """相容 tuple / dict 兩種 DataLoader 格式。"""
        if isinstance(batch, (list, tuple)):
            return batch[1].to(device)
        elif isinstance(batch, dict):
            return batch["data"].to(device)
        raise TypeError(f"[VIM] Unknown batch type: {type(batch)}")

    def _calculate_params(self):
        """計算 null-space NS 與縮放係數 alpha。"""
        feat_dim = self._feature_id_train.shape[1]  # 256

        if self.dim <= 0 or self.dim >= feat_dim:
            raise ValueError(
                f"[VIM] vim_dim={self.dim} is invalid for feature dim={feat_dim}. "
                f"Must be in range [1, {feat_dim - 1}]. "
                f"Recommended: {feat_dim // 4} ~ {feat_dim * 3 // 4}."
            )

        # eigh 回傳由小到大排序，index 0 = 最小特徵值（最小變異方向）
        # ID subspace = 特徵值最大的前 dim 個（index [-dim:] 到最後）
        # Null-space  = 特徵值最小的後 (feat_dim - dim) 個（index [:-dim]）
        ns_vecs = self._eigen_vectors.T[:feat_dim - self.dim]   # (feat_dim-dim, feat_dim)
        self.NS = np.ascontiguousarray(ns_vecs.T)               # (feat_dim, feat_dim-dim)

        vlogit_id  = norm(
            np.matmul(self._feature_id_train - self.u, self.NS), axis=-1
        )
        self.alpha = self._logit_id_train.max(axis=-1).mean() / vlogit_id.mean()

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, classifier, abmil, train_loader,
            task_type: str = "patch", device: str = "cuda"):
        """
        從 ID train set 擬合 VIM 參數。

        Args:
            classifier:   Classifier(in_channel=in_dim, hidden_layer=256)
            abmil:        ABMILPooling
            train_loader: split="train" 的 DataLoader
            task_type:    "patch" or "slide"
            device:       "cuda" or "cpu"
        """
        classifier.eval()
        abmil.eval()

        # 取 fc2 的 weight / bias
        self.w = classifier.fc2.weight.detach().cpu().numpy()   # (C, 256)
        self.b = classifier.fc2.bias.detach().cpu().numpy()     # (C,)

        print("[VIM] Extracting ID training hidden features (fc2 input)...", file=sys.stderr)
        feature_list = []
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="VIM fit", leave=True):
                chief_data = self._unpack_batch(batch, device)
                hidden = self._extract_hidden(classifier, abmil, chief_data)
                feature_list.append(hidden.cpu().numpy())

        self._feature_id_train = np.concatenate(feature_list, axis=0)  # (N, 256)
        self._logit_id_train   = self._feature_id_train @ self.w.T + self.b  # (N, C)

        # Principal anchor: u = -W⁺ b
        self.u = -np.matmul(pinv(self.w), self.b)               # (256,)

        # Covariance & eigen-decomposition
        # eigh 專為實數對稱矩陣設計，保證回傳實數且由小到大排序，比 eig 更穩定
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(self._feature_id_train - self.u)
        self._eig_vals, self._eigen_vectors = np.linalg.eigh(ec.covariance_)

        self._calculate_params()
        self.setup_flag = True
        print(f"[VIM] Fit complete. dim={self.dim}, alpha={self.alpha:.4f}", file=sys.stderr)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, save_path: str):
        """儲存 score() 所需的最小參數集合。"""
        torch.save({
            "w":     self.w,
            "b":     self.b,
            "u":     self.u,
            "NS":    self.NS,
            "alpha": self.alpha,
            "dim":   self.dim,
        }, save_path)
        print(f"[VIM] Parameters saved to {save_path}", file=sys.stderr)

    def load(self, save_path: str):
        """從磁碟載入預先計算好的參數，跳過 train set 掃描。"""
        params = torch.load(save_path, map_location="cpu", weights_only=False)
        self.w          = params["w"]
        self.b          = params["b"]
        self.u          = params["u"]
        self.NS         = params["NS"]
        self.alpha      = params["alpha"]
        self.dim        = params["dim"]
        self.setup_flag = True
        print(f"[VIM] Parameters loaded from {save_path} "
              f"(dim={self.dim}, alpha={self.alpha:.4f})", file=sys.stderr)

    # ------------------------------------------------------------------
    # score
    # ------------------------------------------------------------------

    def score(self, classifier, abmil, chief_data, device: str = "cuda"):
        """
        計算單一 batch 的 VIM OOD score。

        Returns:
            torch.Tensor (float32), shape (B,) — higher = more ID
        """
        if not self.setup_flag:
            raise RuntimeError(
                "[VIM] Parameters not ready. "
                "Run fit_vim.py first to generate vim_params.pt."
            )

        with torch.no_grad():
            hidden = self._extract_hidden(classifier, abmil, chief_data)
            feature = hidden.cpu().numpy()                  # (B, 256)

        logit  = feature @ self.w.T + self.b               # (B, C)
        energy = logsumexp(logit, axis=-1)                  # (B,)
        vlogit = norm(
            np.matmul(feature - self.u, self.NS), axis=-1
        ) * self.alpha                                      # (B,)

        return torch.from_numpy((-vlogit + energy).astype(np.float32))

    # ------------------------------------------------------------------
    # Hyperparameter search helper
    # ------------------------------------------------------------------

    def set_dim(self, dim: int):
        """更新 dim 並重新計算（不需重新 fit）。須在 fit() 同一 session 內呼叫。"""
        if self._feature_id_train is None:
            raise RuntimeError(
                "[VIM] set_dim() requires raw training features. "
                "Call fit() in the same session before set_dim()."
            )
        self.dim = dim
        self._calculate_params()
        print(f"[VIM] dim updated to {self.dim}, alpha={self.alpha:.4f}", file=sys.stderr)