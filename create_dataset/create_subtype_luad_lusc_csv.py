"""
建立 LUAD vs LUSC 亞型分類任務的 4-fold CSV 資料集

任務定義:
  - LUAD : label = 0
  - LUSC : label = 1
  - OOD (選用，--ood_subdir 指定) : label = -1，ood_label = 1，只出現在 test set

Patient ID (檔名前 12 字元, e.g. TCGA-05-4244) 作為 Group，
確保同一病人的切片不會同時出現在 train / val / test。

CSV 內的路徑是相對於 --feature_root 的相對路徑，訓練時再以相同的
feature_root 還原，因此 CSV 可以在不同機器間共用。

Usage:
    python create_dataset/create_subtype_luad_lusc_csv.py --data_type patch --feature_root /path/to/patch_embeddings \
        --ood_subdir 'TCGA-BRCA-FS/CHIEF/20X/pt_files(stain_norm)'
    # 多個 OOD 來源：重複 --ood_subdir；不給 --ood_subdir 則 test set 只有 ID
"""
import argparse
import pandas as pd, numpy as np, os
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# feature_root 底下的子目錄（CHIEF 特徵的預設目錄結構）
DEFAULT_SUBDIRS = {
    'patch': ('TCGA-LUAD-FS/CHIEF/20X/pt_files(stain_norm)',
              'TCGA-LUSC-FS/CHIEF/20X/pt_files(stain_norm)'),
    'slide': ('TCGA-LUAD-FS/CHIEF_WSI/20X/pt_files(stain_norm)',
              'TCGA-LUSC-FS/CHIEF_WSI/20X/pt_files(stain_norm)'),
}
N_FOLDS, SEED = 4, 42

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--data_type', required=True, choices=['patch', 'slide'])
parser.add_argument('--feature_root', required=True,
                    help='特徵根目錄（例如 .../patch_embeddings）；CSV 內路徑相對於此')
parser.add_argument('--luad_subdir', default=None,
                    help='feature_root 底下的 LUAD .pt 目錄（預設依 data_type 決定）')
parser.add_argument('--lusc_subdir', default=None,
                    help='feature_root 底下的 LUSC .pt 目錄（預設依 data_type 決定）')
parser.add_argument('--ood_subdir', action='append', default=[],
                    help='feature_root 底下的 OOD .pt 目錄，只放進 test set；可重複指定多個來源。'
                         '不指定則 test set 只有 ID 樣本')
parser.add_argument('--output_dir', default=None,
                    help='輸出目錄（預設 <repo>/data/4_fold/subtype_LUAD_vs_LUSC_<data_type>）')
args = parser.parse_args()

luad_subdir = args.luad_subdir or DEFAULT_SUBDIRS[args.data_type][0]
lusc_subdir = args.lusc_subdir or DEFAULT_SUBDIRS[args.data_type][1]
OUTPUT_DIR  = args.output_dir or os.path.join(
    REPO_ROOT, 'data', '4_fold', f'subtype_LUAD_vs_LUSC_{args.data_type}')
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(SEED)

def scan(subdir, lbl):
    fs = sorted([f for f in os.listdir(os.path.join(args.feature_root, subdir)) if f.endswith('.pt')])
    return np.array([os.path.join(subdir,f) for f in fs]), np.array([lbl]*len(fs)), np.array([f[:12] for f in fs])

def scan_ood(subdirs):
    paths = []
    for subdir in subdirs:
        fs = sorted([f for f in os.listdir(os.path.join(args.feature_root, subdir)) if f.endswith('.pt')])
        paths += [os.path.join(subdir, f) for f in fs]
    return np.array(paths, dtype=object)

lX, ly, lg = scan(luad_subdir, 0)
sX, sy, sg = scan(lusc_subdir, 1)
oX = scan_ood(args.ood_subdir)
print(f'LUAD={len(lX)}, LUSC={len(sX)}, OOD={len(oX)}'
      f" ({', '.join(args.ood_subdir) or '未指定 --ood_subdir，test set 只有 ID'})")

X = np.concatenate([lX, sX])
y = np.concatenate([ly, sy])
g = np.concatenate([lg, sg])
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
buckets = [idx for _, idx in sgkf.split(X, y, groups=g)]
o_indices = np.arange(len(oX))
np.random.shuffle(o_indices)
ob = np.array_split(o_indices, N_FOLDS)

for i in range(N_FOLDS):
    ti, vi = buckets[i], buckets[(i+1)%N_FOLDS]
    tr = np.concatenate([buckets[j] for j in range(N_FOLDS) if j!=i and j!=(i+1)%N_FOLDS])
    Xt,yt = X[ti],y[ti]; Xv,yv = X[vi],y[vi]; Xr,yr = X[tr],y[tr]; Xo = oX[ob[i]]
    mp = np.concatenate([Xt, Xo])
    ml = np.concatenate([yt, np.full(len(Xo), -1)])
    mo = np.concatenate([np.zeros(len(Xt)), np.ones(len(Xo))])
    df = pd.concat([
        pd.DataFrame({'train':Xr,'train_label':yr,'ood_label':0}),
        pd.DataFrame({'val':Xv,'val_label':yv,'ood_label':0}),
        pd.DataFrame({'test':mp,'test_label':ml,'ood_label':mo}),
    ], ignore_index=True)[['train','train_label','val','val_label','test','test_label','ood_label']]
    df.to_csv(os.path.join(OUTPUT_DIR, f'dataset_fold_{i}.csv'), index=False)
    print(f'Fold {i}: Train={len(Xr)}, Val={len(Xv)}, Test_ID={len(Xt)}, Test_OOD={len(Xo)}')

print('Done! CSV files saved to:', OUTPUT_DIR)
