import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utilmodule.utils import make_parse
from utilmodule.core import train, seed_torch
from torch.utils.data import DataLoader
from dataset.load_datasets import PatchDataset
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


def main(args):
    seed_torch(args.seed)

    # Resolve fold-level save_dir: save_dir/fold_{fold}
    fold = args.fold
    if fold is not None:
        fold_dir = os.path.join(args.save_dir, f"fold_{fold}")
    else:
        # Fallback: use csv filename as folder name (legacy behaviour)
        run_name = args.csv.split("/")[-1].split(".")[0]
        fold_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(fold_dir, exist_ok=True)
    # Store fold_dir on args so core.py can use it
    args.fold_dir = fold_dir

    data_csv_dir = args.csv
    feature_root = args.feature_root

    print(f"Config:  {args.config}")
    print(f"CSV:     {data_csv_dir}")
    print(f"Save:    {fold_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = Classifier(in_channel=args.in_dim, hidden_layer=256).to(device)
    classifier.apply(init_weights)
    abmil = ABMILPooling(in_dim=args.in_dim, hidden_dim=256).to(device)

    train_dataset = PatchDataset(data_csv_dir, feature_root, "train")
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    validation_dataset = PatchDataset(data_csv_dir, feature_root, "val")
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
    test_dataset = PatchDataset(data_csv_dir, feature_root, "test")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # WandB project: from config or fallback to csv-based name
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    if args.use_wandb:
        wandb_project = getattr(args, "wandb_project", None) or (
            args.csv.split("/")[-3] + "_abMIL"
        )
        run_name = getattr(args, "experiment_name", None) or args.csv.split("/")[-1].split(".")[0]
        if fold is not None:
            run_name = f"{run_name}_fold{fold}"
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb.init(project=wandb_project, name=run_name, config=vars(args))


    train(args, classifier=classifier, abmil=abmil,
          train_loader=train_dataloader,
          validation_loader=validation_dataloader,
          test_loader=test_dataloader,
          wandb=wandb)


if __name__ == "__main__":
    args = make_parse()
    main(args)
