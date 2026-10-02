import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utilmodule.utils import make_parse
from utilmodule.core import test, seed_torch
from torch.utils.data import DataLoader
from dataset.load_datasets import PatchDataset
import torch
import torch.nn as nn
from models.classifier import Classifier
from models.ABMIL import ABMILPooling
import pickle
import wandb
from dotenv import load_dotenv


def main(args):
    seed_torch(args.seed)

    data_csv_dir = args.csv
    feature_root = args.feature_root

    # Resolve model directory (test_dir)
    fold = args.fold
    if args.test_dir is None:
        if fold is not None:
            args.test_dir = os.path.join(args.save_dir, f"fold_{fold}")
        else:
            run_name = args.csv.split("/")[-1].split(".")[0]
            args.test_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(args.test_dir, exist_ok=True)

    print(f"Config:    {args.config}")
    print(f"CSV:       {data_csv_dir}")
    print(f"Model dir: {args.test_dir}")

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    if args.use_wandb:
        wandb_project = getattr(args, "wandb_project", None) or (
            args.csv.split("/")[-3] + "_abMIL"
        )
        run_name = getattr(args, "experiment_name", None) or args.csv.split("/")[-1].split(".")[0]
        if fold is not None:
            run_name = f"{run_name}_fold{fold}_test"
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb.init(project=wandb_project, name=run_name, config=vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = Classifier(in_channel=args.in_dim, hidden_layer=256).to(device)
    classifier_weight = torch.load(os.path.join(args.test_dir, "classifier.pth"), map_location=device)
    classifier.load_state_dict(classifier_weight)

    abmil = ABMILPooling(in_dim=args.in_dim, hidden_dim=256).to(device)
    abmil_weight = torch.load(os.path.join(args.test_dir, "abmil.pth"), map_location=device)
    abmil.load_state_dict(abmil_weight)

    test_dataset = PatchDataset(data_csv_dir, feature_root, "test")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    classifier.eval()
    abmil.eval()

    test(args, classifier, abmil, test_dataloader, run_type="test", epoch=0,
         wandb=wandb if args.use_wandb else None, run_time_test=False, see_inside=False)


if __name__ == "__main__":
    args = make_parse()
    main(args)
