"""
fit_vim.py — 預先計算並儲存 VIM 參數（vim_params_dim{dim}.pt）

使用方式：
    python ood_methods/fit_vim.py --config configs/subtype_LUAD_vs_LUSC.yaml --fold 0 \
        --feature_root /path/to/slide_embeddings --vim_dim 128

執行後會在 model_dir 下產生 vim_params_dim{dim}.pt，
之後 ood_test.py --method vim 直接載入，不需重跑 train set。
"""
import os
import argparse
import torch

from common import add_common_args, resolve_args, load_models, make_loader
from vim import VIMPostprocessor


def fit_vim(args, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Model dir : {args.model_dir}")
    print(f"CSV       : {args.csv}")
    print(f"Task type : {args.task_type}")
    print(f"VIM dim   : {args.vim_dim}")

    classifier, abmil = load_models(args, device)
    train_loader = make_loader(args, "train")

    print(f"🔄 Fitting VIM using {len(train_loader)} ID train samples...")

    vim = VIMPostprocessor(dim=args.vim_dim)
    vim.fit(
        classifier=classifier,
        abmil=abmil,
        train_loader=train_loader,
        task_type=args.task_type,
        device=str(device),
    )

    save_path = os.path.join(args.model_dir, f"vim_params_dim{args.vim_dim}.pt")
    vim.save(save_path)
    print(f"✅ VIM parameters saved to {save_path}")


if __name__ == "__main__":
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--vim_dim", type=int, default=128,
                        help="Null-space dimensions to discard (tune 64–512)")
    args = parser.parse_args()
    cfg = resolve_args(parser, args)
    fit_vim(args, cfg)
