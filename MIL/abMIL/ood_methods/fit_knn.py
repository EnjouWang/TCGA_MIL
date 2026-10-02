"""
fit_knn.py — 建立 KNN OOD 的 faiss index（knn_params_K{K}_{hidden|bag}.npz）

使用 ID train set 的 L2-normalised feature：預設為 classifier hidden (256d)，
加 --use_bag_embedding 則改用 bag embedding (in_dim 維)。

使用方式：
    python ood_methods/fit_knn.py --config configs/subtype_LUAD_vs_LUSC.yaml --fold 0 \
        --feature_root /path/to/slide_embeddings --K 50
"""
import os
import argparse
import torch

from common import add_common_args, resolve_args, load_models, make_loader
from knn import KNNPostprocessor


def fit_knn(args, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Model dir : {args.model_dir}")
    print(f"CSV       : {args.csv}")
    print(f"Task type : {args.task_type}")
    print(f"KNN K     : {args.K}")
    print(f"Feature   : {f'bag_embedding ({args.in_dim})' if args.use_bag_embedding else 'hidden/fc2-input (256)'}")

    classifier, abmil = load_models(args, device)
    train_loader = make_loader(args, "train")

    print(f"🔄 Fitting KNN index using {len(train_loader)} ID train samples...")

    knn = KNNPostprocessor(K=args.K, use_bag_embedding=args.use_bag_embedding)
    knn.fit(
        classifier=classifier,
        abmil=abmil,
        train_loader=train_loader,
        task_type=args.task_type,
        device=str(device),
    )

    # npz 副檔名由 save() 自動加上
    feat_tag  = "bag" if args.use_bag_embedding else "hidden"
    save_path = os.path.join(args.model_dir, f"knn_params_K{args.K}_{feat_tag}")
    knn.save(save_path)
    print(f"✅ KNN parameters saved to {save_path}.npz")


if __name__ == "__main__":
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--K", type=int, default=50,
                        help="Number of nearest neighbours (typical: 10~50)")
    parser.add_argument("--use_bag_embedding", action="store_true",
                        help="Use the bag_embedding (in_dim-d) instead of the 256-d hidden feature")
    args = parser.parse_args()
    cfg = resolve_args(parser, args)
    fit_knn(args, cfg)
