"""test_slide.py — Test entry point for slide-level embedding experiments.

Usage:
    python test_slide.py --config configs/subtype_LUAD_vs_LUSC.yaml --fold 0
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..") ))
from utilmodule.utils import make_parse
from utilmodule.core import test, seed_torch
from torch.utils.data import DataLoader
from dataset.load_datasets import SlideDataset
import torch
import wandb
from dotenv import load_dotenv
from models.classifier import Classifier
from models.ABMIL import ABMILPooling
from train_slide import slide_collate_fn   # reuse collate_fn


def main(args):
    seed_torch(args.seed)

    # Resolve model directory (where classifier.pth / abmil.pth were saved)
    fold = args.fold
    if args.test_dir is None:
        if args.save_dir and fold is not None:
            args.test_dir = os.path.join(args.save_dir, f"fold_{fold}")
        else:
            raise ValueError("--test_dir or (--save_dir + --fold) required.")

    print(f"Config:   {args.config}")
    print(f"CSV:      {args.csv}")
    print(f"ModelDir: {args.test_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = args.in_dim

    classifier = Classifier(in_channel=in_dim, hidden_layer=256).to(device)
    abmil      = ABMILPooling(in_dim=in_dim, hidden_dim=256).to(device)

    classifier.load_state_dict(torch.load(os.path.join(args.test_dir, "classifier.pth"), map_location=device))
    abmil.load_state_dict(torch.load(os.path.join(args.test_dir, "abmil.pth"), map_location=device))
    classifier.eval()
    abmil.eval()

    test_dataset = SlideDataset(args.csv, args.feature_root, "test")
    test_loader  = DataLoader(test_dataset, batch_size=1,
                              shuffle=False, collate_fn=slide_collate_fn,
                              num_workers=2, pin_memory=True)

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    if args.use_wandb:
        wandb_project = getattr(args, "wandb_project", None) or "slide_abMIL"
        run_name = getattr(args, "experiment_name", None) or "slide_test"
        if fold is not None:
            run_name = f"{run_name}_fold{fold}"
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb.init(project=wandb_project, name=run_name, config=vars(args))

    # args.csv_saveName is used by core.test() to name the output CSV
    if not hasattr(args, "csv_saveName") or not args.csv_saveName:
        args.csv_saveName = "probability.csv"

    test(args, classifier=classifier, abmil=abmil,
         test_loader=test_loader,
         run_type="test",
         epoch=0,
         wandb=wandb,
         run_time_test=False)


if __name__ == "__main__":
    args = make_parse()
    main(args)
