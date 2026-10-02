"""Shared plumbing for the post-hoc OOD scripts (fit_*.py, get_react_threshold.py,
ood_test.py): argument parsing, config/path resolution, model loading and loaders.

Path rules match training (see utilmodule/utils.py):
  - csv_dir / save_base_dir in the config resolve against the repo root;
  - feature_root comes from --feature_root or the config and is required.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml
from torch.utils.data import DataLoader

from dataset.load_datasets import PatchDataset, SlideDataset
from models.classifier import Classifier
from models.ABMIL import ABMILPooling
from train_slide import slide_collate_fn
from utilmodule.utils import _repo_path


def add_common_args(parser):
    parser.add_argument("--config",       type=str, default=None)
    parser.add_argument("--fold",         type=int, default=None)
    parser.add_argument("--csv",          type=str, default=None,
                        help="Fold CSV (default: <csv_dir>/dataset_fold_<fold>.csv from the config)")
    parser.add_argument("--model_dir",    type=str, default=None,
                        help="Checkpoint dir (default: <save_base_dir>/<experiment_name>/fold_<fold>)")
    parser.add_argument("--feature_root", type=str, default=None,
                        help="Root of the feature .pt files (or config key feature_root)")
    parser.add_argument("--in_dim",       type=int, default=None,
                        help="Feature dimension of the foundation model (config key in_dim; default 768)")
    return parser


def resolve_args(parser, args):
    """Fill csv / model_dir / feature_root / task_type from the config. Returns the cfg dict."""
    cfg = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    if args.model_dir is None:
        save_base = cfg.get("save_base_dir")
        exp_name  = cfg.get("experiment_name")
        if save_base and exp_name and args.fold is not None:
            args.model_dir = os.path.join(_repo_path(save_base), exp_name, f"fold_{args.fold}")
    if args.csv is None:
        csv_dir = cfg.get("csv_dir")
        if csv_dir and args.fold is not None:
            args.csv = os.path.join(_repo_path(csv_dir), f"dataset_fold_{args.fold}.csv")
    if args.feature_root is None:
        args.feature_root = cfg.get("feature_root")

    if args.model_dir is None:
        parser.error("--model_dir is required unless --config and --fold are given.")
    if args.csv is None:
        parser.error("--csv is required unless --config (with csv_dir) and --fold are given.")
    if not args.feature_root:
        parser.error("--feature_root is required (or set feature_root in the config).")

    if args.in_dim is None:
        args.in_dim = int(cfg.get("in_dim", 768))
    args.task_type = cfg.get("task_type", "patch")
    return cfg


def load_models(args, device):
    """Load the trained (classifier, abmil) pair from args.model_dir in eval mode."""
    in_dim, model_dir = args.in_dim, args.model_dir
    classifier = Classifier(in_channel=in_dim, hidden_layer=256).to(device)
    abmil      = ABMILPooling(in_dim=in_dim, hidden_dim=256).to(device)
    classifier.load_state_dict(
        torch.load(os.path.join(model_dir, "classifier.pth"), map_location=device))
    abmil.load_state_dict(
        torch.load(os.path.join(model_dir, "abmil.pth"), map_location=device))
    classifier.eval()
    abmil.eval()
    return classifier, abmil


def make_loader(args, split):
    """batch_size=1, unshuffled loader over one split of the fold CSV."""
    if args.task_type == "slide":
        dataset = SlideDataset(args.csv, args.feature_root, split)
        return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=slide_collate_fn)
    dataset = PatchDataset(args.csv, args.feature_root, split)
    return DataLoader(dataset, batch_size=1, shuffle=False)
