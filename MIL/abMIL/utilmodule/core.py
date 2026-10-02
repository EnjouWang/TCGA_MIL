import torch.nn as nn
import torch.nn.functional as F
import torch
from utilmodule.utils import calculate_metrics, resolve_class_weights
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
import pickle
import os
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import copy


# def test(args, cluster_record, classifier, abmil, test_loader, run_type="test", epoch=0, wandb=None, run_time_test=True, see_inside=False):
def test(args, classifier, abmil, test_loader, run_type="test", epoch=0, wandb=None, run_time_test=True, see_inside=False, return_loss=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Reuse the class weights train() resolved, so val loss is on the same scale as
    # train loss (early stopping compares val loss across epochs). None -> unweighted,
    # which is what every caller that never sets `class_weights` gets.
    _cw = getattr(args, "class_weight_tensor", None)
    if _cw is not None:
        _cw = _cw.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=_cw, reduction="sum")

    label_list = []
    Y_prob_list = []
    ood_labels_list = []  # 用來收集每一筆的 OOD 標籤
    filenames_list = []   # 用來收集每一筆的檔名
    
    loss_total = 0
    ce_loss = 0
    
    with torch.no_grad():
        for idx, (_, chief_data, label, name, ood_label) in enumerate(tqdm(test_loader)):

            # 將資料送入 GPU
            chief_data = chief_data.to(device)
            label = label.to(device).long()
            
            # 模型推論
            bag_embedding, atten = abmil(chief_data)
            logits = classifier(bag_embedding)
            loss = compute_loss(logits, label, atten, criterion)
            probs = F.softmax(logits , dim=1)

            # 收集機率與標籤
            Y_prob_list.append(probs.detach().cpu())
            label_list.append(label.detach().cpu())
            
            # 收集檔名 — 支援 batch_size >= 1
            filenames_list.extend(list(name))
            ood_labels_list.extend(torch.as_tensor(ood_label).reshape(-1).tolist())

            loss_total += loss["loss"].item()
            ce_loss += loss["ce_loss"].item()

        # --- 迴圈結束，整理數據 ---

        targets = np.asarray(torch.cat(label_list, dim=0).detach().cpu().numpy()).reshape(-1)
        probs = np.asarray(torch.cat(Y_prob_list, dim=0).detach().cpu().numpy())

        # 計算指標時只保留 ID 樣本 (label >= 0)，排除 OOD (label=-1)
        id_mask = targets >= 0
        precision, recall, f1, auc, accuracy = calculate_metrics(targets[id_mask], probs[id_mask])
        preds = np.argmax(probs, axis=1)
        correct = (preds == targets).astype(int)

        # 1. 訓練/驗證階段：上傳 WandB 曲線
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,   # shared x-axis so val/* overlays train/* per epoch
                f"{run_type}/precision": precision,
                f"{run_type}/recall": recall,
                f"{run_type}/f1": f1,
                f"{run_type}/auc": auc,
                f"{run_type}/acc": accuracy,
                f"{run_type}/loss": loss_total/len(test_loader),
                f"{run_type}/ce_loss": ce_loss/len(test_loader),
            })
            
        # 2. 儲存詳細 CSV
        if not run_time_test:
            # 確保所有欄位長度一致
            df = pd.DataFrame({
                'filename': filenames_list,    # 加入檔名
                'label': targets,
                'pred': preds,
                'correct': correct,
                'prob': probs[:, 1],           # 取出 class 1 的機率
                'ood_label': ood_labels_list,
            })
            
            save_path = os.path.join(args.test_dir, args.csv_saveName)
            df.to_csv(save_path, index=False)
            print(f"✅ CSV saved to {save_path}")

            if args.use_wandb:
                wandb.log({
                    "test/ID_score_dist": wandb.Histogram(df[df['ood_label'] == 0]['prob'].tolist()),
                    "test/OOD_score_dist": wandb.Histogram(df[df['ood_label'] == 1]['prob'].tolist()),
                })

    # return_loss lets the training loop early-stop on validation LOSS without
    # changing test()'s default return (other callers still get just `auc`).
    if return_loss:
        return auc, loss_total / len(test_loader)
    return auc



def compute_loss(logits, label, attn, criterion):
    """
    logits: [B, num_classes]
    label:  [B]   (-1 for OOD samples that have no task label)

    Only computes CE loss over ID samples (label >= 0).
    Returns zero loss if all samples in the batch are OOD.

    `criterion` is built with reduction='sum'; this function divides by the number of
    valid samples. That is deliberate and NOT equivalent to reduction='mean' when
    class weights are used: torch's 'mean' normalises by the SUM OF WEIGHTS, i.e.
    sum(w_i*l_i)/sum(w_i), which at batch_size=1 collapses to (w_y*l)/w_y = l — the
    weight cancels out exactly and class weighting silently becomes a no-op. Dividing
    by N instead gives sum(w_i*l_i)/N, so the weighting survives at any batch size.
    With no weights the two are identical, so unweighted tasks are unaffected.
    """
    loss = {}
    valid_mask = label >= 0
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        loss["ce_loss"] = torch.tensor(0.0, device=logits.device, requires_grad=False)
    else:
        loss["ce_loss"] = criterion(logits[valid_mask], label[valid_mask]) / n_valid
    loss["loss"] = loss["ce_loss"]
    return loss

def normalized_attention_entropy(attn, eps=1e-12):
    """
    Normalized entropy in [0, 1]
    """
    if attn.dim() == 2:
        attn = attn.squeeze(0)

    N = attn.numel()
    attn = attn.clamp(min=eps)

    entropy = -torch.sum(attn * torch.log(attn))
    entropy_norm = entropy / torch.log(torch.tensor(N, device=attn.device, dtype=attn.dtype))
    return entropy_norm

def train(args, classifier, abmil, train_loader, validation_loader, test_loader=None, wandb=None):

    # Use fold_dir if set by train.py (new config-driven flow), else fall back to legacy path
    if hasattr(args, 'fold_dir') and args.fold_dir:
        save_dir = args.fold_dir
    else:
        run_name = f"{args.csv.split('/')[-1].split('.')[0]}"
        save_dir = os.path.join(args.save_dir, run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier.train()
    abmil.train()
    # for p in abmil.parameters():
    #     p.requires_grad = False

    # weight_decay is config-driven; default 1e-4 preserves the previous hardcoded value
    # for every task that does not set it.
    wd = getattr(args, "weight_decay", None)
    if wd is None:
        wd = 1e-4
    optimizer = torch.optim.Adam(
        list(abmil.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=wd
    )
    print(f"weight_decay = {wd}")
    # --- class weighting (opt-in via config `class_weights`; default: unweighted) ---
    # Counters the degenerate "always predict the majority class" optimum on
    # imbalanced tasks, which leaves precision/recall/f1 at 0 under the fixed 0.5
    # threshold. Weights are derived from the TRAIN split only.
    class_weight = resolve_class_weights(
        getattr(args, "class_weights", None),
        labels=getattr(train_loader.dataset, "labels", None),
        num_classes=args.n_classes if hasattr(args, "n_classes") else 2,
    )
    if class_weight is not None:
        class_weight = class_weight.to(device)
        print(f"Class weights ({getattr(args, 'class_weights', None)}): "
              f"{[round(float(x), 4) for x in class_weight]}")
    # Stash so test() weights the VALIDATION loss identically — early stopping now
    # monitors val loss, so a weighted train loss and an unweighted val loss would
    # be optimising two different objectives and stop at the wrong epoch.
    args.class_weight_tensor = class_weight

    criterion = torch.nn.CrossEntropyLoss(weight=class_weight, reduction="sum")

    none_epoch = 0
    # Early-stopping metric is config-switchable (`early_stop_metric`): 'val_loss'
    # (default, the tie-robust choice on tiny val sets) or 'val_auroc' (branch experiment
    # — selects the epoch with the best validation ranking instead of the lowest loss).
    # Both are unified into a HIGHER-IS-BETTER `score`: val_auroc uses AUROC directly,
    # val_loss is negated, so the same strict '>' comparison and "a tie does NOT reset
    # patience" semantics cover both branches.
    es_metric = getattr(args, "early_stop_metric", "val_loss") or "val_loss"
    if es_metric not in ("val_loss", "val_auroc"):
        raise ValueError(f"early_stop_metric must be 'val_loss' or 'val_auroc', got {es_metric!r}")
    print(f"Early stopping monitors: {es_metric} (patience={args.patience})")
    best_score = float('-inf')
    # Track the best BOTH models. abmil must be snapshotted too: it keeps training past
    # the best epoch, so the live object is the LAST abmil, not the best. Pairing
    # best_classifier with a stale abmil at the final test() gave a mismatched model.
    # Defaults cover num_epochs==0.
    best_classifier = copy.deepcopy(classifier)
    best_abmil      = copy.deepcopy(abmil)

    # pkl_path = os.path.join(args.chief_feature_dir, "cluster_record_spatialleiden.pkl")
    # with open(pkl_path, "rb") as f:
    #     cluster_record = pickle.load(f)

    print("Training !!!")
    for idx, epoch in enumerate(range(args.num_epochs)):

        Y_prob_list = []
        label_list = []
        loss_total = 0
        ce_loss = 0
        atten_entropy_loss = 0

        for ide, (_, chief_data, label, name, _ood) in enumerate(tqdm(train_loader)):

            # group = cluster_record[name[0]]["groups"]
            # group_means = [
            #     g.mean(dim=0, keepdim=True)  # (1, D)
            #     for g in group
            # ]
            # group_feature = torch.cat(group_means, dim=0).unsqueeze(0).to(device)
            chief_data = chief_data.to(device)
            label = label.to(device).long()
            # Clear gradients from the previous step. Without this they accumulate
            # across every batch and epoch, which collapses the classifier (P(class 1)
            # never reaches 0.5 -> precision/recall/f1 all 0).
            optimizer.zero_grad()
            bag_embedding, attention = abmil(chief_data)
            logits = classifier(bag_embedding)
            # loss = compute_loss(logits, label, logits, criterion)
            loss = compute_loss(logits, label, attention, criterion)
            loss["loss"].backward()
            optimizer.step()
            probs = F.softmax(logits , dim=1)

            # record
            Y_prob_list.append(probs.detach().cpu())
            label_list.append(label.detach().cpu())
            loss_total += loss["loss"].item()
            ce_loss += loss["ce_loss"].item()
            # atten_entropy_loss += loss["atten_entropy_loss"].item()
            # print(loss["atten_entropy_loss"].item())


        # train record
        train_targets = np.asarray(torch.cat(label_list, dim=0).detach().cpu().numpy()).reshape(-1)
        train_probs = np.asarray(torch.cat(Y_prob_list, dim=0).detach().cpu().numpy())
        train_precision, train_recall, train_f1, train_auc, train_accuracy = calculate_metrics(train_targets, train_probs)
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/precision": train_precision,
                "train/recall": train_recall,
                "train/f1": train_f1,
                "train/auc": train_auc,
                "train/acc": train_accuracy,
                "train/loss": loss_total/len(train_loader),
                "train/ce_loss": ce_loss/len(train_loader),
                # "train/atten_entropy_loss": atten_entropy_loss/len(train_loader),
            })
        

        # val — both metrics are always computed; `es_metric` picks which one drives
        # early stopping. val_loss surfaces overfitting before val AUC does; val_auroc
        # optimises ranking directly (but ties on tiny val sets — see project memory).
        val_auc, val_loss = test(args, classifier, abmil, validation_loader,
                                 run_type="val", epoch=epoch, wandb=wandb, return_loss=True)
        score = val_auc if es_metric == "val_auroc" else -val_loss

        # Strict improvement ('>'): a tie does NOT reset patience, so training actually
        # stops on a plateau instead of resetting every equal epoch. A non-finite score
        # (e.g. a degenerate val AUROC) counts as no improvement.
        if np.isfinite(score) and score > best_score:
            best_score = score
            none_epoch = 0
            # snapshot + save the best pair (classifier AND abmil together)
            best_classifier = copy.deepcopy(classifier)
            best_abmil      = copy.deepcopy(abmil)
            torch.save(classifier.state_dict(), os.path.join(save_dir, f"classifier.pth"))
            torch.save(abmil.state_dict(), os.path.join(save_dir, f"abmil.pth"))
        else:
            none_epoch += 1
            if none_epoch >= args.patience:
                print(f"Break at epoch {epoch}.")
                break

    # Final test on the best-val-loss PAIR (was best_classifier + stale live abmil).
    test(args, best_classifier, best_abmil, test_loader, run_type="test", epoch=0, wandb=wandb)



def seed_torch(seed=2021):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 
