"""
get_react_threshold.py — 計算 ReAct 的 bag-embedding 截斷閾值

使用 ID val set 的 bag embedding 取第 --percentile 百分位，輸出 react_threshold.txt 到 model_dir。

使用方式：
    python ood_methods/get_react_threshold.py --config configs/subtype_LUAD_vs_LUSC.yaml --fold 0 \
        --feature_root /path/to/slide_embeddings
"""
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from common import add_common_args, resolve_args, load_models, make_loader


def get_threshold(args, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 載入模型
    _, abmil = load_models(args, device)

    # 2. 載入 ID Validation Data
    loader = make_loader(args, "val")

    print(f"🔄 Calculating ReAct threshold using {len(loader)} ID validation samples...")
    print(f"   Task Type: {args.task_type}, CSV: {args.csv}")

    activation_log = []

    # 3. 收集 Bag Embeddings
    with torch.no_grad():
        for _, chief_data, label, name, *others in tqdm(loader):
            chief_data = chief_data.to(device)

            # ABMIL 的輸出: bag_embedding, attention
            bag_embedding, _ = abmil(chief_data)

            # 轉成 numpy 並存起來
            activation_log.append(bag_embedding.cpu().numpy().flatten())

    # 4. 合併並計算分位數
    all_activations = np.concatenate(activation_log)
    threshold = np.percentile(all_activations, args.percentile)

    print(f"\n📊 Stats:")
    print(f"   Min: {all_activations.min():.4f}")
    print(f"   Max: {all_activations.max():.4f}")
    print(f"   Mean: {all_activations.mean():.4f}")
    print(f"✅ ReAct Threshold (p={args.percentile}): {threshold:.4f}")

    save_path = os.path.join(args.model_dir, "react_threshold.txt")
    with open(save_path, "w") as f:
        f.write(str(threshold))
    print(f"💾 Threshold saved to {save_path}")

    return threshold


if __name__ == "__main__":
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument('--percentile', type=float, default=90, help="Percentile for threshold (e.g., 90)")
    args = parser.parse_args()
    cfg = resolve_args(parser, args)
    get_threshold(args, cfg)
