"""
建立 LUAD 腫瘤偵測任務的 4-fold CSV 資料集

任務定義:
  - LUAD 無腫瘤 (Normal, sample code 10~19) : label = 0
  - LUAD 有腫瘤 (Tumor,  sample code 01~09) : label = 1
  - OOD (選用，--ood_subdir 指定)          : label = -1，ood_label = 1，只出現在 test set

Patient ID (前三段, e.g. TCGA-4B-A93V) 作為 Group，
確保同一病人的切片不會同時出現在 train / val / test。

CSV 內的路徑是相對於 --feature_root 的相對路徑，訓練時再以相同的
feature_root 還原，因此 CSV 可以在不同機器間共用。

Usage:
    python create_dataset/create_detection_luad.py --data_type patch --feature_root /path/to/patch_embeddings \
        --ood_subdir 'TCGA-BRCA-FS/CHIEF/20X/pt_files(stain_norm)'
    # 多個 OOD 來源：重複 --ood_subdir；不給 --ood_subdir 則 test set 只有 ID
"""

import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# feature_root 底下的 LUAD 子目錄（CHIEF 特徵的預設目錄結構）
DEFAULT_LUAD_SUBDIR = {
    'patch': 'TCGA-LUAD-FS/CHIEF/20X/pt_files(stain_norm)',
    'slide': 'TCGA-LUAD-FS/CHIEF_WSI/20X/pt_files(stain_norm)',
}

N_FOLDS = 4
SEED    = 42


# ── 解析 TCGA 檔名 ──────────────────────────────────────────────────────────────
def parse_tcga_filename(filename: str):
    """
    從 TCGA 檔名解析 label 和 patient ID。

    範例: TCGA-4B-A93V-01A-01-TSA.pt
      - Patient ID  : TCGA-4B-A93V   (前三段，用於 group 切分)
      - Sample code : 01             (第四段前兩碼)
          01~09 → Tumor  (label = 1)
          10~19 → Normal (label = 0)

    Returns:
        patient_id : str
        label      : int (0 or 1)

    Raises:
        ValueError : 若格式不符或 sample code 超出預期範圍
    """
    parts = os.path.basename(filename).replace('.pt', '').split('-')
    if len(parts) < 4:
        raise ValueError(f"無法解析檔名: {filename}")

    patient_id  = '-'.join(parts[:3])          # TCGA-4B-A93V
    sample_code = int(parts[3][:2])            # '01A' → 01

    if 1 <= sample_code <= 9:
        label = 1   # Tumor
    elif 10 <= sample_code <= 19:
        label = 0   # Normal
    else:
        raise ValueError(f"未預期的 sample code {sample_code:02d} in {filename}")

    return patient_id, label


# ── 掃描資料夾 ──────────────────────────────────────────────────────────────────
def scan_luad(feature_root: str, subdir: str):
    """掃描 feature_root/subdir，依檔名解析 label 與 patient group。
    回傳的路徑相對於 feature_root。"""
    paths, labels, groups = [], [], []

    for fname in sorted(os.listdir(os.path.join(feature_root, subdir))):
        if not fname.endswith('.pt'):
            continue
        rel_path = os.path.join(subdir, fname)
        try:
            patient_id, label = parse_tcga_filename(fname)
        except ValueError as e:
            print(f"⚠️  跳過: {e}")
            continue
        paths.append(rel_path)
        labels.append(label)
        groups.append(patient_id)

    return np.array(paths), np.array(labels), np.array(groups)


def scan_ood(feature_root: str, subdirs):
    """依序掃描各 OOD 目錄，回傳相對於 feature_root 的路徑（無 OOD 時為空陣列）。"""
    paths = []
    for subdir in subdirs:
        files = sorted([f for f in os.listdir(os.path.join(feature_root, subdir)) if f.endswith('.pt')])
        paths += [os.path.join(subdir, f) for f in files]
    return np.array(paths, dtype=object)


# ── 主程式 ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_type', required=True, choices=['patch', 'slide'])
    parser.add_argument('--feature_root', required=True,
                        help='特徵根目錄（例如 .../patch_embeddings）；CSV 內路徑相對於此')
    parser.add_argument('--luad_subdir', default=None,
                        help='feature_root 底下的 LUAD .pt 目錄（預設依 data_type 決定）')
    parser.add_argument('--ood_subdir', action='append', default=[],
                        help='feature_root 底下的 OOD .pt 目錄，只放進 test set；可重複指定多個來源。'
                             '不指定則 test set 只有 ID 樣本')
    parser.add_argument('--output_dir', default=None,
                        help='輸出目錄（預設 <repo>/data/4_fold/detection_LUAD_<data_type>）')
    return parser.parse_args()


def main():
    args = parse_args()
    luad_subdir = args.luad_subdir or DEFAULT_LUAD_SUBDIR[args.data_type]
    output_dir  = args.output_dir or os.path.join(
        REPO_ROOT, 'data', '4_fold', f'detection_LUAD_{args.data_type}')
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(SEED)

    # 讀取資料
    luad_X, luad_y, luad_g = scan_luad(args.feature_root, luad_subdir)
    ood_X                   = scan_ood(args.feature_root, args.ood_subdir)

    tumor_count  = (luad_y == 1).sum()
    normal_count = (luad_y == 0).sum()
    print(f"LUAD Tumor (1) = {tumor_count}")
    print(f"LUAD Normal(0) = {normal_count}")
    print(f"OOD       (-1) = {len(ood_X)}  ({', '.join(args.ood_subdir) or '未指定 --ood_subdir，test set 只有 ID'})")
    print(f"LUAD 總計       = {len(luad_X)}")

    # StratifiedGroupKFold：按 label 分層、按 patient group 避免 data leakage
    sgkf    = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    buckets = [idx for _, idx in sgkf.split(luad_X, luad_y, groups=luad_g)]

    # 將 OOD 樣本平均分配到各 fold 的 test set
    ood_indices = np.arange(len(ood_X))
    np.random.shuffle(ood_indices)
    ood_buckets = np.array_split(ood_indices, N_FOLDS)

    # 建立各 fold 的 CSV
    for i in range(N_FOLDS):
        # test  = bucket i
        # val   = bucket (i+1) % N_FOLDS
        # train = 剩餘兩個 bucket
        test_idx = buckets[i]
        val_idx  = buckets[(i + 1) % N_FOLDS]
        train_idx = np.concatenate([
            buckets[j] for j in range(N_FOLDS)
            if j != i and j != (i + 1) % N_FOLDS
        ])

        # LUAD splits
        X_train, y_train = luad_X[train_idx], luad_y[train_idx]
        X_val,   y_val   = luad_X[val_idx],   luad_y[val_idx]
        X_test,  y_test  = luad_X[test_idx],  luad_y[test_idx]

        # OOD for this fold's test set
        X_ood = ood_X[ood_buckets[i]]

        # Test set = LUAD (ID) + OOD
        test_paths  = np.concatenate([X_test, X_ood])
        test_labels = np.concatenate([y_test, np.full(len(X_ood), -1)])
        test_ood    = np.concatenate([np.zeros(len(X_test), dtype=int),
                                      np.ones(len(X_ood),  dtype=int)])

        df = pd.concat([
            pd.DataFrame({'train':       X_train,
                          'train_label': y_train,
                          'ood_label':   0}),
            pd.DataFrame({'val':         X_val,
                          'val_label':   y_val,
                          'ood_label':   0}),
            pd.DataFrame({'test':        test_paths,
                          'test_label':  test_labels,
                          'ood_label':   test_ood}),
        ], ignore_index=True)[['train', 'train_label',
                                'val',   'val_label',
                                'test',  'test_label', 'ood_label']]

        out_path = os.path.join(output_dir, f'dataset_fold_{i}.csv')
        df.to_csv(out_path, index=False)

        print(f"\nFold {i}:")
        print(f"  Train : {len(X_train)}  "
              f"(Tumor={( y_train==1).sum()}, Normal={(y_train==0).sum()})")
        print(f"  Val   : {len(X_val)}  "
              f"(Tumor={( y_val==1).sum()},   Normal={(y_val==0).sum()})")
        print(f"  Test  : ID={len(X_test)} "
              f"(Tumor={( y_test==1).sum()},  Normal={(y_test==0).sum()})  "
              f"OOD={len(X_ood)}")

    print("\n✅ Done! CSV files saved to:", output_dir)


if __name__ == '__main__':
    main()