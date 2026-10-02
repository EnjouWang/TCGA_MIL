"""
Residual OOD Detection
Reference: Zaeemzadeh et al.,
           "Out-of-Distribution Detection Using Union of 1-Dimensional Subspaces", CVPR 2021
Adapted from: https://github.com/remic-othr/OpenMIBOOD
Copyright (c) 2021 Jingkang Yang. Licensed under the MIT License;
see THIRD_PARTY_NOTICES.md at the repo root for the full license text.

與 VIM 的關係：
  VIM   score = -vlogit * alpha + energy  (null-space norm + energy 混合)
  Residual score = -norm(null-space projection)  (純幾何，無 energy 項，無 alpha 縮放)

fit 使用 train set（與 VIM/KNN 相同），避免 val set 同時作為 fit set 和評估 set
造成資料洩漏。
"""

import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import norm, pinv
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm


class ResidualPostprocessor:
    """
    Residual OOD scorer for AB-MIL models.

    原理：
      1. 從 ID train set 的 feature 擬合 covariance，取前 dim 個主成分為 ID subspace
      2. 計算 test feature 在 null-space（剩餘方向）的投影 norm
      3. score = -norm  →  norm 越小（殘差越小）→ 越 ID

    Args:
        dim (int): ID subspace 的維度（保留特徵值最大的前 dim 個主成分）。
                   null-space = 256 - dim 個維度。
                   典型範圍：64 ~ 192。
    """

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.setup_flag = False

        self.w  = None   # (C, 256)  fc2.weight
        self.b  = None   # (C,)      fc2.bias
        self.u  = None   # (256,)    principal anchor
        self.NS = None   # (256, 256-dim)  null-space projection

        self._feature_id_val  = None   # fit features（train set），供 _calculate_params 使用
        self._eig_vals        = None
        self._eigen_vectors   = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_hidden(classifier, abmil, chief_data):
        """取得 fc2 的輸入 feature：relu(fc1(bag_embedding))，shape (B, 256)。"""
        bag_embedding, _ = abmil(chief_data)
        hidden = F.relu(classifier.fc1(bag_embedding))
        return hidden

    @staticmethod
    def _unpack_batch(batch, device: str):
        """相容 tuple / dict 兩種 DataLoader 格式。"""
        if isinstance(batch, (list, tuple)):
            return batch[1].to(device)
        elif isinstance(batch, dict):
            return batch["data"].to(device)
        raise TypeError(f"[Residual] Unknown batch type: {type(batch)}")

    def _calculate_params(self):
        """
        計算 null-space NS。

        eigh 回傳由小到大：
          index [-dim:] = 特徵值最大的 dim 個  → ID subspace
          index [:-dim] = 特徵值最小的其餘個  → null-space (NS)
        """
        feat_dim = self._feature_id_val.shape[1]  # 256

        if self.dim <= 0 or self.dim >= feat_dim:
            raise ValueError(
                f"[Residual] dim={self.dim} is invalid for feature dim={feat_dim}. "
                f"Must be in range [1, {feat_dim - 1}]."
            )

        ns_vecs = self._eigen_vectors.T[:feat_dim - self.dim]  # (feat_dim-dim, feat_dim)
        self.NS = np.ascontiguousarray(ns_vecs.T)              # (feat_dim, feat_dim-dim)

    # ------------------------------------------------------------------
    # fit  ── 使用 train set（避免 val set 同時作為 fit set 與評估 set）
    # ------------------------------------------------------------------

    def fit(self, classifier, abmil, fit_loader,
            task_type: str = "patch", device: str = "cuda"):
        """
        從 ID train set 擬合 Residual 參數。

        使用 train set（而非 val set）以避免當 val set 作為 evaluator 評估集時
        產生資料洩漏。null-space 由 train features 的主成分決定。

        Args:
            classifier:  Classifier(in_channel=in_dim, hidden_layer=256)
            abmil:       ABMILPooling
            fit_loader:  split="train" 的 DataLoader
            task_type:   "patch" or "slide"
            device:      "cuda" or "cpu"
        """
        classifier.eval()
        abmil.eval()

        self.w = classifier.fc2.weight.detach().cpu().numpy()  # (C, 256)
        self.b = classifier.fc2.bias.detach().cpu().numpy()    # (C,)

        print("[Residual] Extracting ID train hidden features (fc2 input)...")
        feature_list = []
        with torch.no_grad():
            for batch in tqdm(fit_loader, desc="Residual fit", leave=True):
                chief_data = self._unpack_batch(batch, device)
                hidden = self._extract_hidden(classifier, abmil, chief_data)
                feature_list.append(hidden.cpu().numpy())

        self._feature_id_val = np.concatenate(feature_list, axis=0)  # (N, 256)

        # Principal anchor: u = -W⁺ b
        self.u = -np.matmul(pinv(self.w), self.b)              # (256,)

        # Covariance & eigen-decomposition（eigh：實數對稱矩陣，穩定，由小到大）
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(self._feature_id_val - self.u)
        self._eig_vals, self._eigen_vectors = np.linalg.eigh(ec.covariance_)

        self._calculate_params()
        self.setup_flag = True
        print(f"[Residual] Fit complete. dim={self.dim}, "
              f"NS shape={self.NS.shape}, N_train={len(self._feature_id_val)}")

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, save_path: str):
        """儲存 score() 所需的最小參數集合。"""
        torch.save({
            "w":    self.w,
            "b":    self.b,
            "u":    self.u,
            "NS":   self.NS,
            "dim":  self.dim,
        }, save_path)
        print(f"[Residual] Parameters saved to {save_path}")

    def load(self, save_path: str):
        """從磁碟載入預先計算好的參數。"""
        params = torch.load(save_path, map_location="cpu", weights_only=False)
        self.w          = params["w"]
        self.b          = params["b"]
        self.u          = params["u"]
        self.NS         = params["NS"]
        self.dim        = params["dim"]
        self.setup_flag = True
        print(f"[Residual] Parameters loaded from {save_path} (dim={self.dim})")

    # ------------------------------------------------------------------
    # score
    # ------------------------------------------------------------------

    def score(self, classifier, abmil, chief_data, device: str = "cuda"):
        """
        計算單一 batch 的 Residual OOD score。

        Returns:
            torch.Tensor (float32), shape (B,) — higher = more ID
        """
        if not self.setup_flag:
            raise RuntimeError(
                "[Residual] Parameters not ready. "
                "Run fit_residual.py first to generate residual_params.pt."
            )

        with torch.no_grad():
            hidden = self._extract_hidden(classifier, abmil, chief_data)
            feature = hidden.cpu().numpy()                     # (B, 256)

        # null-space 殘差的 norm，取負號使 higher = more ID
        residual = norm(np.matmul(feature - self.u, self.NS), axis=-1)  # (B,)
        scores   = -residual
        return torch.from_numpy(scores.astype(np.float32))

    # ------------------------------------------------------------------
    # Hyperparameter search helper
    # ------------------------------------------------------------------

    def set_dim(self, dim: int):
        """
        更新 dim 並重新計算 NS（不需重新 fit）。
        須在 fit() 同一 session 內呼叫。
        """
        if self._feature_id_val is None:
            raise RuntimeError(
                "[Residual] set_dim() requires raw fit features. "
                "Call fit() in the same session before set_dim()."
            )
        self.dim = dim
        self._calculate_params()
        print(f"[Residual] dim updated to {self.dim}, NS shape={self.NS.shape}")