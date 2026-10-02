"""
ood_test.py — Post-hoc OOD evaluation on a trained fold.

Scores every test slide with one post-hoc method and reports AUROC / FPR95 of
ID (ood_label=0) vs OOD (ood_label=1). Methods other than msp / energy need their
fit step first (see run_experiment.sh --mode ood, which runs it automatically).

Usage:
    python ood_test.py --config configs/subtype_LUAD_vs_LUSC.yaml --fold 0 \
        --feature_root /path/to/slide_embeddings --method maha
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import argparse
import numpy as np
import pandas as pd
import torch
import wandb
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from ood_methods.common import add_common_args, resolve_args, load_models, make_loader
from ood_methods.methods import PostHocOOD
from ood_methods.vim import VIMPostprocessor
from ood_methods.knn import KNNPostprocessor
from ood_methods.residual import ResidualPostprocessor


def calculate_ood_metrics(id_scores, ood_scores):
    """
    AUROC and FPR95.
    Convention: higher score = more ID-like (consistent across all methods).
    """
    y_true   = np.concatenate([np.ones(len(id_scores)),  np.zeros(len(ood_scores))])
    y_scores = np.concatenate([id_scores, ood_scores])
    auroc    = roc_auc_score(y_true, y_scores)

    # FPR at TPR=0.95: threshold set at 5th percentile of ID scores
    threshold = np.percentile(id_scores, 5)
    fpr95     = np.sum(ood_scores > threshold) / len(ood_scores)
    return auroc, fpr95


def main(args, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.method == "react" and args.threshold is None:
        thresh_file = os.path.join(args.model_dir, "react_threshold.txt")
        if not os.path.exists(thresh_file):
            raise ValueError("ReAct requires --threshold or react_threshold.txt in model_dir. "
                             "Run ood_methods/get_react_threshold.py first.")
        with open(thresh_file) as f:
            args.threshold = float(f.read().strip())
        print(f"Loaded ReAct threshold: {args.threshold}")

    print(f"Model dir : {args.model_dir}")
    print(f"CSV       : {args.csv}")
    print(f"Method    : {args.method}")
    print(f"Task type : {args.task_type}")

    classifier, abmil = load_models(args, device)
    loader = make_loader(args, "test")

    # ── Load auxiliary parameters ────────────────────────────────────────────────
    ood_engine         = PostHocOOD(classifier, abmil, device)
    maha_params        = None
    vim_processor      = None
    knn_processor      = None
    residual_processor = None

    if args.method == "maha":
        maha_path = os.path.join(args.model_dir, "maha_params.pt")
        if not os.path.exists(maha_path):
            raise ValueError("maha_params.pt not found. Run ood_methods/fit_maha.py first.")
        maha_params = torch.load(maha_path, map_location=device)

    if args.method == "vim":
        vim_path = os.path.join(args.model_dir, f"vim_params_dim{args.vim_dim}.pt")
        if not os.path.exists(vim_path):
            raise ValueError(
                f"vim_params_dim{args.vim_dim}.pt not found in {args.model_dir}. "
                f"Run fit_vim.py first:\n"
                f"  python ood_methods/fit_vim.py --config <cfg> --fold <n> --vim_dim {args.vim_dim}"
            )
        vim_processor = VIMPostprocessor(dim=args.vim_dim)
        vim_processor.load(vim_path)

    if args.method == "knn":
        feat_tag  = "bag" if args.knn_use_bag else "hidden"
        knn_path  = os.path.join(args.model_dir, f"knn_params_K{args.knn_K}_{feat_tag}.npz")
        if not os.path.exists(knn_path):
            bag_flag = " --use_bag_embedding" if args.knn_use_bag else ""
            raise ValueError(
                f"knn_params_K{args.knn_K}_{feat_tag}.npz not found in {args.model_dir}. "
                f"Run fit_knn.py first:\n"
                f"  python ood_methods/fit_knn.py --config <cfg> --fold <n> --K {args.knn_K}{bag_flag}"
            )
        knn_processor = KNNPostprocessor(K=args.knn_K)
        knn_processor.load(knn_path)

    if args.method == "residual":
        residual_path = os.path.join(args.model_dir, f"residual_params_dim{args.residual_dim}.pt")
        if not os.path.exists(residual_path):
            raise ValueError(
                f"residual_params_dim{args.residual_dim}.pt not found in {args.model_dir}. "
                f"Run fit_residual.py first:\n"
                f"  python ood_methods/fit_residual.py --config <cfg> --fold <n> --dim {args.residual_dim}"
            )
        residual_processor = ResidualPostprocessor(dim=args.residual_dim)
        residual_processor.load(residual_path)

    # ── Inference ────────────────────────────────────────────────────────────────
    id_scores_list  = []
    ood_scores_list = []
    results         = []

    for _, chief_data, label, name, ood_label in tqdm(loader):
        ood_label  = int(torch.as_tensor(ood_label).reshape(-1)[0])
        chief_data = chief_data.to(device)

        if args.method == "msp":
            score = ood_engine.msp(chief_data)
        elif args.method == "energy":
            score = ood_engine.energy(chief_data, T=args.temperature)
        elif args.method == "react":
            score = ood_engine.react(chief_data, threshold=args.threshold,
                                     energy_T=args.temperature)
        elif args.method == "maha":
            score = ood_engine.mahalanobis_pp(chief_data, maha_params)
        elif args.method == "vim":
            score = ood_engine.vim(chief_data, vim_processor)
        elif args.method == "knn":
            score = ood_engine.knn(chief_data, knn_processor)
        elif args.method == "residual":
            score = ood_engine.residual(chief_data, residual_processor)

        score_val = score.item()
        if ood_label == 0:
            id_scores_list.append(score_val)
        else:
            ood_scores_list.append(score_val)

        results.append({"filename": name[0], "ood_label": ood_label, "score": score_val})

    # ── Metrics ──────────────────────────────────────────────────────────────────
    if not id_scores_list or not ood_scores_list:
        raise ValueError(
            f"The test split of {args.csv} has {len(id_scores_list)} ID and "
            f"{len(ood_scores_list)} OOD slides; OOD evaluation needs both. "
            "Regenerate the split with --ood_subdir (see create_dataset/).")
    auroc, fpr95 = calculate_ood_metrics(
        np.array(id_scores_list), np.array(ood_scores_list))
    print(f"\nOOD Detection — AUROC: {auroc:.4f}  FPR95: {fpr95:.4f}")

    # ── Save CSV ─────────────────────────────────────────────────────────────────
    df            = pd.DataFrame(results)
    save_filename = f"ood_result_{args.method}.csv"
    if args.method == "react":
        save_filename = f"ood_result_react_thr{args.threshold:.2f}.csv"
    elif args.method == "vim":
        save_filename = f"ood_result_vim_dim{args.vim_dim}.csv"
    elif args.method == "knn":
        feat_tag = "bag" if args.knn_use_bag else "hidden"
        save_filename = f"ood_result_knn_K{args.knn_K}_{feat_tag}.csv"
    elif args.method == "residual":
        save_filename = f"ood_result_residual_dim{args.residual_dim}.csv"
    save_path = os.path.join(args.model_dir, save_filename)
    df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

    # ── WandB ────────────────────────────────────────────────────────────────────
    if args.use_wandb:
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
        exp_name      = cfg.get("experiment_name", "exp")
        fold_str      = str(args.fold) if args.fold is not None else "?"
        wandb_project = cfg.get("wandb_project", exp_name + "_OOD")
        run_name      = f"{exp_name}_fold{fold_str}_{args.method}"

        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb.init(project=wandb_project, name=run_name, config=vars(args))
        wandb.log({
            "ood/auroc":  auroc,
            "ood/fpr95":  fpr95,
            "ood/method": args.method,
        })
        wandb.log({
            "ID_dist":  wandb.Histogram(id_scores_list),
            "OOD_dist": wandb.Histogram(ood_scores_list),
        })


if __name__ == "__main__":
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--method",       type=str,   default="msp",
                        choices=["msp", "energy", "react", "maha",
                                 "vim", "knn", "residual"])
    parser.add_argument("--threshold",    type=float, default=None,
                        help="ReAct clip threshold (default: react_threshold.txt in model_dir)")
    parser.add_argument("--temperature",  type=float, default=1.0)
    parser.add_argument("--vim_dim",      type=int,   default=128,
                        help="Must match the dim used in fit_vim.py")
    parser.add_argument("--knn_K",        type=int,   default=50,
                        help="Must match the K used in fit_knn.py")
    parser.add_argument("--knn_use_bag",  action="store_true",
                        help="Use bag_embedding (in_dim-d) instead of hidden (256d) for KNN")
    parser.add_argument("--residual_dim", type=int,   default=128,
                        help="Must match the dim used in fit_residual.py")
    parser.add_argument("--use_wandb",    action="store_true")
    args = parser.parse_args()
    cfg = resolve_args(parser, args)
    main(args, cfg)
