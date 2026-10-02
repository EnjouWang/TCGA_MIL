"""train_slide.py — Training entry point for slide-level embedding experiments.

Uses SlideDataset + slide_collate_fn (wraps [D] embedding as [1,1,D])
so that ABMILPooling and core.train() work unchanged.

Usage:
    python train_slide.py --config configs/subtype_LUAD_vs_LUSC.yaml --fold 0
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..") ))
from utilmodule.utils import make_parse
from utilmodule.core import train, seed_torch
from torch.utils.data import DataLoader
from dataset.load_datasets import SlideDataset
import torch
import torch.nn as nn
import torch.nn.init as init
import wandb
from dotenv import load_dotenv
from models.classifier import Classifier
from models.ABMIL import ABMILPooling


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, a=0.01)
        if m.bias is not None:
            init.constant_(m.bias, 0)


def slide_collate_fn(batch):
    """Wrap [D] slide embeddings as [B,1,D] to match core.py expectations.

    core.py train loop unpacks:  (_, chief_data, label, name, ood_label)
    where chief_data is [B, N, D].  We use N=1 (single slide = 1-patch bag).
    """
    embeddings, labels, paths, ood_labels = zip(*batch)
    chief_data   = torch.stack(embeddings, dim=0).unsqueeze(1)   # [B,1,D]
    label_tensor = torch.tensor(labels,    dtype=torch.long)
    ood_tensor   = torch.tensor(ood_labels,dtype=torch.long)
    coords_placeholder = [None] * len(embeddings)
    return coords_placeholder, chief_data, label_tensor, paths, ood_tensor


def main(args):
    seed_torch(args.seed)

    # Resolve fold-level save directory
    fold = args.fold
    if fold is not None:
        fold_dir = os.path.join(args.save_dir, f"fold_{fold}")
    else:
        run_name = args.csv.split("/")[-1].split(".")[0]
        fold_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(fold_dir, exist_ok=True)
    args.fold_dir = fold_dir

    print(f"Config:  {args.config}")
    print(f"CSV:     {args.csv}")
    print(f"Save:    {fold_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim     = args.in_dim
    batch_size = args.batch_size

    classifier = Classifier(in_channel=in_dim, hidden_layer=256).to(device)
    classifier.apply(init_weights)
    abmil = ABMILPooling(in_dim=in_dim, hidden_dim=256).to(device)

    train_dataset = SlideDataset(args.csv, args.feature_root, "train")
    val_dataset   = SlideDataset(args.csv, args.feature_root, "val")
    test_dataset  = SlideDataset(args.csv, args.feature_root, "test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  collate_fn=slide_collate_fn,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, collate_fn=slide_collate_fn,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=1,
                              shuffle=False, collate_fn=slide_collate_fn,
                              num_workers=2, pin_memory=True)

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    if args.use_wandb:
        wandb_project = getattr(args, "wandb_project", None) or "slide_abMIL"
        run_name = getattr(args, "experiment_name", None) or "slide_exp"
        if fold is not None:
            run_name = f"{run_name}_fold{fold}"
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb.init(project=wandb_project, name=run_name, config=vars(args))

    train(args, classifier=classifier, abmil=abmil,
          train_loader=train_loader,
          validation_loader=val_loader,
          test_loader=test_loader,
          wandb=wandb)


if __name__ == "__main__":
    args = make_parse()
    main(args)
