"""
fit_residual.py — 預先計算並儲存 Residual 參數（residual_params_dim{dim}.pt）

Residual 與 VIM 共用 null-space 投影，但只用投影 norm（無 energy 項、無 alpha）。
Fit 使用 ID train set。

使用方式：
    python ood_methods/fit_residual.py --config configs/subtype_LUAD_vs_LUSC.yaml --fold 0 \
        --feature_root /path/to/slide_embeddings --dim 128
"""
import os
import argparse
import torch

from common import add_common_args, resolve_args, load_models, make_loader
from residual import ResidualPostprocessor


def fit_residual(args, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Model dir : {args.model_dir}")
    print(f"CSV       : {args.csv}")
    print(f"Task type : {args.task_type}")
    print(f"Residual dim : {args.dim}")

    classifier, abmil = load_models(args, device)
    train_loader = make_loader(args, "train")

    print(f"🔄 Fitting Residual using {len(train_loader)} ID train samples...")

    residual = ResidualPostprocessor(dim=args.dim)
    residual.fit(
        classifier=classifier,
        abmil=abmil,
        fit_loader=train_loader,
        task_type=args.task_type,
        device=str(device),
    )

    save_path = os.path.join(args.model_dir, f"residual_params_dim{args.dim}.pt")
    residual.save(save_path)
    print(f"✅ Residual parameters saved to {save_path}")


if __name__ == "__main__":
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--dim", type=int, default=128,
                        help="ID subspace dimensions (typical: 64~192)")
    args = parser.parse_args()
    cfg = resolve_args(parser, args)
    fit_residual(args, cfg)
