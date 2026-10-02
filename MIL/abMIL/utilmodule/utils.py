import pandas as pd
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import os
import shutil
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix
import heapq
import statistics
import yaml


# Repo root (<repo>/MIL/abMIL/utilmodule/utils.py -> <repo>). Relative paths in a
# config (csv_dir, save_base_dir) resolve against it, so runs don't depend on cwd.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _repo_path(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def load_config(config_path):
    """Load a YAML config file and return it as a dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_class_weights(spec, labels=None, num_classes=2):
    """Resolve a ``class_weights`` spec into a float tensor of per-class CE weights.

    Accepted forms (all optional — the default is no weighting, so every task that
    does not set ``class_weights`` behaves exactly as before):

      None / "none" / "" / False   -> None  (unweighted CrossEntropyLoss)
      "balanced"                   -> inverse frequency  w_c = N / (K * n_c),
                                      computed from ``labels`` (the train split)
      [0.58, 3.51] / "0.58,3.51"   -> explicit per-class weights, one per class

    Returns a torch.FloatTensor of length ``num_classes``, or None.
    """
    if spec is None or spec is False:
        return None
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in ("", "none", "null", "false", "off"):
            return None
        if s == "balanced":
            spec = "balanced"
        else:
            # explicit list given on the CLI, e.g. --class_weights 0.58,3.51
            try:
                spec = [float(x) for x in spec.replace(" ", "").split(",") if x != ""]
            except ValueError:
                raise ValueError(f"Cannot parse --class_weights {spec!r}")

    if spec == "balanced":
        if labels is None:
            raise ValueError("class_weights='balanced' needs the train labels")
        y = np.asarray(labels).reshape(-1)
        y = y.astype(int)
        counts = np.bincount(y, minlength=num_classes).astype(float)
        if (counts == 0).any():
            missing = np.where(counts == 0)[0].tolist()
            raise ValueError(
                f"class_weights='balanced': class(es) {missing} absent from the train "
                f"split (counts={counts.tolist()}); cannot invert frequency."
            )
        w = len(y) / (num_classes * counts)
        return torch.tensor(w, dtype=torch.float)

    w = [float(x) for x in spec]
    if len(w) != num_classes:
        raise ValueError(
            f"class_weights has {len(w)} entries but the task has {num_classes} classes"
        )
    return torch.tensor(w, dtype=torch.float)


def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file. CLI args override config values.")

    # --- Paths ---
    parser.add_argument("--feature_root", type=str, default=None,
                        help="Root directory of the feature .pt files; CSV paths are relative "
                             "to it. Required (CLI or config key feature_root).")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to a single CSV (overrides config csv_dir + fold).")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Override save directory (default: save_base_dir/experiment_name).")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Override model/result directory for testing.")

    # --- Training ---
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--num_epochs", default=None, type=int)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--in_dim", type=int, default=None,
                        help="Feature dimension of the foundation model (config key in_dim; default 768).")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Slide-level batch size (config key batch_size; default 4). "
                             "Patch-level training always uses one bag per step.")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold index (0-3). Used with config csv_dir to build CSV path.")
    parser.add_argument("--class_weights", type=str, default=None,
                        help="Per-class CE weights: 'none' (default), 'balanced' "
                             "(inverse frequency from the train split), or an explicit "
                             "comma-separated list e.g. '0.58,3.51'.")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Adam weight decay (L2). Default 1e-4 when unset.")
    parser.add_argument("--early_stop_metric", type=str, default=None,
                        choices=["val_loss", "val_auroc"],
                        help="Metric that drives early stopping / checkpoint selection: "
                             "'val_loss' (default, tie-robust on tiny val sets) or "
                             "'val_auroc' (branch experiment). Config key: early_stop_metric.")

    # --- Output ---
    parser.add_argument("--csv_saveName", type=str, default="probability.csv")

    # --- Device ---
    parser.add_argument("--cpu", action="store_true",
                        help="Train on CPU on purpose (bypasses the GPU fail-fast check).")

    # --- WandB ---
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")

    args = parser.parse_args()

    # If a config file is provided, load and apply values for unset CLI args
    if args.config is not None:
        cfg = load_config(args.config)
        _apply_config(args, cfg)

    if not args.feature_root:
        parser.error("--feature_root is required (or set feature_root in the config).")

    # Apply hardcoded defaults for anything still None
    if args.seed is None:
        args.seed = 42
    if args.num_epochs is None:
        args.num_epochs = 300
    if args.lr is None:
        args.lr = 0.00001
    if args.patience is None:
        args.patience = 20
    if args.early_stop_metric is None:
        args.early_stop_metric = "val_loss"   # preserve existing behaviour by default
    if args.in_dim is None:
        args.in_dim = 768
    if args.batch_size is None:
        args.batch_size = 4

    return args


def _apply_config(args, cfg):
    """Apply YAML config values. CLI-provided values take priority."""
    key_map = {
        "feature_root": "feature_root",
        "lr":       "lr",
        "patience": "patience",
        "num_epochs": "num_epochs",
        "seed":     "seed",
        "class_weights": "class_weights",
        "weight_decay": "weight_decay",
        "early_stop_metric": "early_stop_metric",
        "in_dim":   "in_dim",
        "batch_size": "batch_size",
    }
    for cfg_key, arg_key in key_map.items():
        if cfg_key in cfg and getattr(args, arg_key) is None:
            setattr(args, arg_key, cfg[cfg_key])

    save_base = cfg.get("save_base_dir", None)
    if save_base:
        save_base = _repo_path(save_base)
    exp_name  = cfg.get("experiment_name", None)

    if args.save_dir is None and save_base and exp_name:
        args.save_dir = os.path.join(save_base, exp_name)

    if args.test_dir is None and save_base and exp_name and args.fold is not None:
        args.test_dir = os.path.join(save_base, exp_name, f"fold_{args.fold}")

    csv_dir = cfg.get("csv_dir", None)
    if args.csv is None and csv_dir is not None and args.fold is not None:
        args.csv = os.path.join(_repo_path(csv_dir), f"dataset_fold_{args.fold}.csv")

    args.experiment_name = cfg.get("experiment_name", "")
    args.wandb_project   = cfg.get("wandb_project", "")
    args.task            = cfg.get("task", "")


def calculate_metrics(targets, probs):
    threshold = 0.5
    predictions = (probs[:, 1] >= threshold).astype(int)
    precision = precision_score(targets, predictions)
    recall    = recall_score(targets, predictions)
    f1        = f1_score(targets, predictions)
    auc       = roc_auc_score(targets, probs[:, 1])
    accuracy  = accuracy_score(targets, predictions)
    return precision, recall, f1, auc, accuracy


def pred_label_process(memory, true_label, cls_pred):
    T = len(memory.results_dict)
    pred = [results_dict["Y_prob"][0][true_label].item() for results_dict in memory.results_dict]
    topk = heapq.nlargest(min(T, 5), pred)
    max_pred = topk[0]
    avg_top3 = statistics.mean(topk[:3])
    avg_top5 = statistics.mean(topk[:5])
    if true_label == 1:
        final_pred = (max_pred + avg_top3 + avg_top5 + cls_pred) / 4
    else:
        final_pred = cls_pred
    return final_pred


def cat_msg2cluster_group(x_groups, msg_tokens):
    x_groups_cated = []
    for x in x_groups:
        x = x.unsqueeze(dim=0)
        try:
            temp = torch.cat((msg_tokens, x), dim=2)
        except Exception as e:
            print("Error when cat msg tokens to sub-bags")
        x_groups_cated.append(temp)
    return x_groups_cated
