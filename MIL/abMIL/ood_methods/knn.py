"""
KNN OOD Detection
Reference: Sun et al., "Out-of-Distribution Detection with Deep Nearest Neighbors", ICML 2022
Adapted from: https://github.com/remic-othr/OpenMIBOOD
Copyright (c) 2021 Jingkang Yang. Licensed under the MIT License;
see THIRD_PARTY_NOTICES.md at the repo root for the full license text.
"""

import numpy as np
import torch
import torch.nn.functional as F
import faiss
from tqdm import tqdm


# L2 正規化，與原始論文一致
normalizer = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)


class KNNPostprocessor:
    """
    KNN OOD scorer for AB-MIL models.

    原理：將 test sample 的 feature 與 ID train set 的 feature 做 KNN 搜尋，
    取第 K 個鄰居的距離作為 OOD score。
    距離越遠 → 越 OOD（score 越低）；距離越近 → 越 ID（score 越高）。

    score = -D_K  (負第K近距離，higher = more ID)

    與 VIM 的關係：
      - VIM 在 feature subspace 中定義幾何距離
      - KNN 直接在整個 feature 空間做最近鄰搜尋，無需假設分布形狀

    feature 空間選擇：
      與 VIM 一致，使用 fc2 的輸入（relu(fc1(bag_embedding))，256維），
      這是模型學到的語義表示空間，比原始 bag_embedding 更有區分力。

    Args:
        K (int): 取第 K 個鄰居的距離。K 越大越平滑，K 越小越敏感。
                 典型值：10~50。
        use_bag_embedding (bool):
            True  → 使用 abmil 輸出的 bag_embedding（in_dim 維），與 Maha 一致。
            False → 使用 relu(fc1(bag_embedding))（256維），與 VIM/Residual 一致。
            預設 False（256維）。若結果不理想可嘗試 True。
    """

    def __init__(self, K: int = 50, use_bag_embedding: bool = False):
        self.K = K
        self.use_bag_embedding = use_bag_embedding
        self.setup_flag = False
        self.activation_log = None  # (N, 256) 正規化後的 train features
        self.index = None           # faiss index

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_feature(self, classifier, abmil, chief_data):
        """
        取得用於 KNN 搜尋的 feature。
          use_bag_embedding=False（預設）: relu(fc1(bag_embedding))，256維，與 VIM/Residual 一致
          use_bag_embedding=True         : bag_embedding，in_dim 維，與 Maha 一致
        """
        bag_embedding, _ = abmil(chief_data)
        if self.use_bag_embedding:
            return bag_embedding
        return F.relu(classifier.fc1(bag_embedding))

    @staticmethod
    def _unpack_batch(batch, device: str):
        """相容 tuple / dict 兩種 DataLoader 格式。"""
        if isinstance(batch, (list, tuple)):
            return batch[1].to(device)
        elif isinstance(batch, dict):
            return batch["data"].to(device)
        raise TypeError(f"[KNN] Unknown batch type: {type(batch)}")

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, classifier, abmil, train_loader,
            task_type: str = "patch", device: str = "cuda"):
        """
        從 ID train set 建立 faiss KNN index。

        Args:
            classifier:   Classifier(in_channel=in_dim, hidden_layer=256)
            abmil:        ABMILPooling
            train_loader: split="train" 的 DataLoader
            task_type:    "patch" or "slide"
            device:       "cuda" or "cpu"
        """
        classifier.eval()
        abmil.eval()

        print("[KNN] Extracting ID training hidden features (fc2 input)...")
        feature_list = []
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="KNN fit", leave=True):
                chief_data = self._unpack_batch(batch, device)
                feature = self._extract_feature(classifier, abmil, chief_data)
                feature_list.append(feature.cpu().numpy())

        features = np.concatenate(feature_list, axis=0)        # (N, 256)
        self.activation_log = normalizer(features).astype(np.float32)

        feat_dim = self.activation_log.shape[1]
        self.index = faiss.IndexFlatL2(feat_dim)
        self.index.add(self.activation_log)

        self.setup_flag = True
        print(f"[KNN] Fit complete. N={len(self.activation_log)}, K={self.K}, "
              f"feat_dim={feat_dim}")

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, save_path: str):
        """
        儲存 KNN index 所需的資料。
        faiss index 本身不能直接用 torch.save，
        改存 activation_log（numpy array）+ K，load 時重建 index。
        """
        np.savez(save_path,
                 activation_log=self.activation_log,
                 K=np.array(self.K),
                 use_bag_embedding=np.array(self.use_bag_embedding))
        print(f"[KNN] Parameters saved to {save_path}.npz")

    def load(self, save_path: str):
        """從磁碟載入並重建 faiss index。"""
        data = np.load(save_path, allow_pickle=False)
        self.activation_log     = data["activation_log"].astype(np.float32)
        self.K                  = int(data["K"])
        self.use_bag_embedding  = bool(data["use_bag_embedding"])

        feat_dim = self.activation_log.shape[1]
        self.index = faiss.IndexFlatL2(feat_dim)
        self.index.add(self.activation_log)

        self.setup_flag = True
        print(f"[KNN] Parameters loaded from {save_path}. "
              f"N={len(self.activation_log)}, K={self.K}, feat_dim={feat_dim}")

    # ------------------------------------------------------------------
    # score
    # ------------------------------------------------------------------

    def score(self, classifier, abmil, chief_data, device: str = "cuda"):
        """
        計算單一 batch 的 KNN OOD score。

        Returns:
            torch.Tensor (float32), shape (B,) — higher = more ID
        """
        if not self.setup_flag:
            raise RuntimeError(
                "[KNN] Parameters not ready. "
                "Run fit_knn.py first to generate knn_params.npz."
            )

        with torch.no_grad():
            feature = self._extract_feature(classifier, abmil, chief_data)
            feature = feature.cpu().numpy()

        feature_normed = normalizer(feature).astype(np.float32)
        D, _ = self.index.search(feature_normed, self.K)       # D: (B, K)

        # 取第 K 個鄰居距離，取負號使 higher = more ID
        kth_dist = -D[:, -1]                                    # (B,)
        return torch.from_numpy(kth_dist.astype(np.float32))