"""
fit_maha.py — 擬合 Mahalanobis++ 參數（class means + shared precision matrix）

使用 ID train set 的 L2-normalised bag embedding，輸出 maha_params.pt 到 model_dir。

使用方式：
    python ood_methods/fit_maha.py --config configs/subtype_LUAD_vs_LUSC.yaml --fold 0 \
        --feature_root /path/to/slide_embeddings
"""
import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import add_common_args, resolve_args, load_models, make_loader


def fit_mahalanobis(args, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, abmil = load_models(args, device)
    loader = make_loader(args, "train")

    print(f"🔄 Fitting Mahalanobis++ using {len(loader)} ID Train samples...")

    all_features = []
    all_labels = []

    with torch.no_grad():
        for _, chief_data, label, name, *others in tqdm(loader):
            chief_data = chief_data.to(device)
            bag_embedding, _ = abmil(chief_data)

            # [Mahalanobis++ 核心]: 強制投影到單位球面
            z = F.normalize(bag_embedding, p=2, dim=1)

            all_features.append(z.cpu())
            all_labels.append(label.cpu())

    all_features = torch.cat(all_features, dim=0) # [N, D]
    all_labels = torch.cat(all_labels, dim=0)     # [N]

    # 篩選掉標籤為 -1 的異常資料
    valid_mask = all_labels >= 0
    all_features = all_features[valid_mask]
    all_labels = all_labels[valid_mask]
    classes = torch.unique(all_labels).long()

    # 計算各類別平均 (Means) 與共享共變異數矩陣 (Shared Covariance)
    class_means = []
    cov_sum = 0

    for c in classes:
        class_features = all_features[all_labels == c]
        mean_c = class_features.mean(dim=0)
        class_means.append(mean_c)

        # 減去平均值
        centered = class_features - mean_c
        # 累加共變異數
        cov_sum += torch.mm(centered.t(), centered)

    class_means = torch.stack(class_means) # [C, D]

    # 樣本共變異數 (除以 N - C)
    shared_cov = cov_sum / (all_features.shape[0] - len(classes))
    # 加入微小對角值防止奇異矩陣 (Singular matrix)
    shared_cov += torch.eye(shared_cov.shape[0]) * 1e-6
    # 取反矩陣 (Precision Matrix)
    precision_matrix = torch.linalg.inv(shared_cov)

    save_path = os.path.join(args.model_dir, "maha_params.pt")
    torch.save({'class_means': class_means, 'precision_matrix': precision_matrix}, save_path)
    print(f"✅ Mahalanobis++ parameters saved to {save_path}")


if __name__ == "__main__":
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    cfg = resolve_args(parser, args)
    fit_mahalanobis(args, cfg)
