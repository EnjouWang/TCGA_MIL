
# TCGA_MIL

Attention-based multiple instance learning (AB-MIL) for two TCGA lung tasks on
pre-computed pathology foundation-model features. The shipped configs use
[CHIEF](https://github.com/hms-dbmi/CHIEF); any other foundation model works
too (see [Using another foundation model](#using-another-foundation-model)).

| Task                     | Classes                 | Source                |
| ------------------------ | ----------------------- | --------------------- |
| `detection_LUAD`       | Normal (0) vs Tumor (1) | TCGA-LUAD             |
| `subtype_LUAD_vs_LUSC` | LUAD (0) vs LUSC (1)    | TCGA-LUAD + TCGA-LUSC |

Each task runs at two feature levels:

- **patch**: a bag of patch embeddings per slide, pooled by AB-MIL
- **slide**: one slide-level embedding per slide

The repo covers the 4-fold patient-level data split, training, testing and
**post-hoc OOD detection**. Out-of-distribution (OOD) slides from one or more
sources you choose (e.g. TCGA-BRCA) are added to every test fold. A trained
model is never shown OOD data; the post-hoc methods score its outputs/features
afterwards.

Feature extraction is not included; the pipeline starts from pre-computed
feature `.pt` files.

---

## Repository structure

```
TCGA_MIL/
├── create_dataset/
│   ├── create_detection_luad.py        # 4-fold split for detection_LUAD
│   └── create_subtype_luad_lusc_csv.py # 4-fold split for subtype_LUAD_vs_LUSC
├── MIL/abMIL/
│   ├── configs/                 # one YAML per task × feature level
│   ├── dataset/load_datasets.py # PatchDataset (patch), SlideDataset (slide)
│   ├── models/                  # ABMILPooling, Classifier
│   ├── utilmodule/              # training/testing loop, arg parsing, metrics
│   ├── train.py / test.py       # patch-level entry points
│   ├── train_slide.py / test_slide.py  # slide-level entry points
│   ├── ood_test.py              # post-hoc OOD evaluation (AUROC / FPR95)
│   ├── ood_methods/             # post-hoc OOD scorers and their fit scripts
│   ├── run_experiment.sh        # launcher: runs all folds for a config
│   └── .env.example             # template for the WandB API key
├── data/4_fold/                 # generated split CSVs (gitignored)
├── experiments/                 # checkpoints and predictions (gitignored)
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

Install PyTorch for your CUDA version first ([instructions](https://pytorch.org/get-started/locally/)),
then the rest:

```bash
pip install -r requirements.txt
```

Tested with Python 3.10 and torch 2.10.0 + CUDA 12.8. The scripts use a GPU
when one is available and fall back to CPU otherwise.

### 2. Configure WandB (`.env`)

Experiment logging goes to [Weights &amp; Biases](https://wandb.ai). The API key
is read from `MIL/abMIL/.env`:

```bash
cp MIL/abMIL/.env.example MIL/abMIL/.env
# edit MIL/abMIL/.env and set WANDB_API_KEY=<your key>
```

| Variable          | Required                         | Description                                                          |
| ----------------- | -------------------------------- | -------------------------------------------------------------------- |
| `WANDB_API_KEY` | When running with`--use_wandb` | Your key from[https://wandb.ai/authorize](https://wandb.ai/authorize) |

- The scripts load `.env` from `MIL/abMIL/` (next to `train.py`), whatever
  directory you run them from.
- `run_experiment.sh` always passes `--use_wandb`, so the launcher needs the
  key. To run without WandB, see [Running without WandB](#running-without-wandb).
- `.env` is gitignored. Do not commit it.

The WandB project for each run comes from the config's `wandb_project` field.

### 3. Prepare the features

The data does not live in the repo. Put the feature files anywhere and point
the code at them with a **feature root** directory, passed at runtime as
`--feature_root` (or set as `feature_root` in a config). Every path in the split
CSVs is relative to this root.

**File format.** One `.pt` file per slide, holding a float `torch.Tensor`:

| Level | Tensor shape | Notes |
|---|---|---|
| patch | `[N_patches, D]` | one row per patch; `N_patches` varies per slide |
| slide | `[D]` or `[1, D]` | one slide-level embedding |

`D` is the feature dimension of your foundation model (768 for CHIEF) and must
match `in_dim` in the config. Patch coordinates are not needed.

**File names** must start with the TCGA barcode, e.g.
`TCGA-05-4244-01A-01-BS1.<uuid>.pt`. The split scripts read the patient ID
(first 12 characters) and, for `detection_LUAD`, the tumor/normal sample code
from it.

**Directory layout.** Each class needs its own directory of `.pt` files under
the feature root; the directory names are up to you. The split scripts default
to the CHIEF layout below, and `--luad_subdir` / `--lusc_subdir` /
`--ood_subdir` point them anywhere else (paths relative to the feature root).

```
<patch_feature_root>/                       # e.g. /data/patch_embeddings
├── TCGA-LUAD-FS/CHIEF/20X/pt_files(stain_norm)/<slide>.pt     # [N_patches, 768]
├── TCGA-LUSC-FS/CHIEF/20X/pt_files(stain_norm)/<slide>.pt
└── TCGA-BRCA-FS/CHIEF/20X/pt_files(stain_norm)/<slide>.pt     # OOD source (optional, any name)

<slide_feature_root>/                       # e.g. /data/slide_embeddings
├── TCGA-LUAD-FS/CHIEF_WSI/20X/pt_files(stain_norm)/<slide>.pt # [768] or [1, 768]
├── TCGA-LUSC-FS/CHIEF_WSI/20X/pt_files(stain_norm)/<slide>.pt
└── TCGA-BRCA-FS/CHIEF_WSI/20X/pt_files(stain_norm)/<slide>.pt # OOD source (optional, any name)
```

`detection_LUAD` only needs the LUAD directory; `subtype_LUAD_vs_LUSC` needs
LUAD and LUSC. The OOD directory has no default (see Step 1).

### Using another foundation model

Nothing in the model or OOD code is tied to CHIEF. To use another foundation
model (e.g. UNI, Virchow2, CONCH):

1. **Extract features** into `.pt` files in the format above (patch:
   `[N_patches, D]`; slide: `[D]`). If your extractor writes another format
   (e.g. features inside `.h5`), convert it to one `.pt` per slide.
2. **Create the splits** with your directories:
   ```bash
   python create_dataset/create_subtype_luad_lusc_csv.py --data_type patch \
       --feature_root /data/uni_patch \
       --luad_subdir TCGA-LUAD/pt_files --lusc_subdir TCGA-LUSC/pt_files \
       --ood_subdir TCGA-BRCA/pt_files \
       --output_dir data/4_fold/subtype_LUAD_vs_LUSC_patch_uni
   ```
   Use a separate `--output_dir` per model so splits for different models do
   not overwrite each other.
3. **Copy a config** (e.g. `configs/subtype_LUAD_vs_LUSC_patch.yaml`) and set:
   - `in_dim`: your model's feature dimension, e.g. 1024 for UNI, 2560 for Virchow2
   - `csv_dir`: the `--output_dir` from step 2
   - `experiment_name` / `wandb_project`: new names, so checkpoints and logs
     stay separate
   - `task_type`: `patch` or `slide`, matching the features
4. **Run** training, testing and OOD detection as usual with the new config.
   All scripts read `in_dim` from the config; `--in_dim` on the command line
   overrides it.

All OOD methods work for any `D`. Checkpoints are tied to the `in_dim` they
were trained with, so always evaluate with the same config.

---

## Usage

All commands run from the repo root unless noted otherwise.

### Step 1: Create the 4-fold splits

```bash
# detection_LUAD, with TCGA-BRCA as the OOD source
python create_dataset/create_detection_luad.py --data_type patch --feature_root /path/to/patch_embeddings \
    --ood_subdir 'TCGA-BRCA-FS/CHIEF/20X/pt_files(stain_norm)'
python create_dataset/create_detection_luad.py --data_type slide --feature_root /path/to/slide_embeddings \
    --ood_subdir 'TCGA-BRCA-FS/CHIEF_WSI/20X/pt_files(stain_norm)'

# subtype_LUAD_vs_LUSC, same arguments
python create_dataset/create_subtype_luad_lusc_csv.py --data_type patch --feature_root /path/to/patch_embeddings \
    --ood_subdir 'TCGA-BRCA-FS/CHIEF/20X/pt_files(stain_norm)'
python create_dataset/create_subtype_luad_lusc_csv.py --data_type slide --feature_root /path/to/slide_embeddings \
    --ood_subdir 'TCGA-BRCA-FS/CHIEF_WSI/20X/pt_files(stain_norm)'
```

`--ood_subdir` chooses the OOD source and has **no default**:

- Any directory of `.pt` files under the feature root works. OOD slides need
  no labels.
- Repeat the flag to combine sources, e.g.
  `--ood_subdir 'TCGA-BRCA-FS/...' --ood_subdir 'TCGA-KIRC-FS/...'`.
- Without it, the test set is ID only. Training and testing work as usual,
  but `ood_test.py` stops with an error because there is nothing to detect.

The ID part of the split does not depend on the OOD choice.

Each run writes `dataset_fold_{0..3}.csv` to
`data/4_fold/<task>_<data_type>/` by default (override with `--output_dir`).
The paths in the CSVs are relative to the feature root, so the same CSVs work
on any machine that has the same feature files.

**Splitting scheme.** Slides are grouped by patient and split with
`StratifiedGroupKFold` (4 buckets, `random_state=42`), so no patient appears in
more than one of train/val/test. For fold *i*: test = bucket *i*,
val = bucket *(i+1) mod 4*, train = the other two buckets.

**OOD samples.** All OOD slides (from every `--ood_subdir`, in the given
order) are shuffled (`np.random.seed(42)`) and divided into 4 equal parts;
part *i* is added to the test set of fold *i*. They never appear in train or
val.

**Labels.**

- `detection_LUAD`: from the TCGA sample-type code (4th barcode field).
  `01`–`09` = Tumor (1), `10`–`19` = Normal (0); other codes are skipped.
- `subtype_LUAD_vs_LUSC`: from the source directory. LUAD = 0, LUSC = 1.
- OOD slides: `test_label = -1`, `ood_label = 1`. All ID slides have
  `ood_label = 0`.

**CSV format.** Columns: `train, train_label, val, val_label, test, test_label, ood_label`.
The three splits are stacked vertically, so each row fills only one pair of
split columns plus `ood_label`.

For reference, with the original feature set the splits contain:

| Task                     | Class 0    | Class 1   | Total slides |
| ------------------------ | ---------- | --------- | ------------ |
| `detection_LUAD`       | 242 Normal | 822 Tumor | 1064         |
| `subtype_LUAD_vs_LUSC` | 1064 LUAD  | 1097 LUSC | 2161         |

With TCGA-BRCA as the OOD source, each task also gets 1939 OOD slides
(484–485 per test fold).

The split depends on the exact set of `.pt` files found. Different files give
different folds.

### Step 2: Train

```bash
cd MIL/abMIL
bash run_experiment.sh --config configs/subtype_LUAD_vs_LUSC_patch.yaml --mode train \
    --feature_root /path/to/patch_embeddings
```

This trains folds 0–3 one after another. Use `--folds "0 1"` to run only some
folds. The config's `task_type` (`patch` / `slide`) decides whether
`train.py` or `train_slide.py` is used.

| Config                                      | `task_type` | Feature root     |
| ------------------------------------------- | ------------- | ---------------- |
| `configs/detection_LUAD_patch.yaml`       | patch         | patch embeddings |
| `configs/detection_LUAD_slide.yaml`       | slide         | slide embeddings |
| `configs/subtype_LUAD_vs_LUSC_patch.yaml` | patch         | patch embeddings |
| `configs/subtype_LUAD_vs_LUSC.yaml`       | slide         | slide embeddings |

Training uses Adam and early stopping on **validation loss**: the checkpoint
with the lowest val loss is kept, and training stops after `patience` epochs
without improvement. At the end of training the best checkpoint is evaluated on
the test split. OOD slides (`label = -1`) are excluded from the loss and from
all classification metrics.

### Step 3: Test

```bash
cd MIL/abMIL
bash run_experiment.sh --config configs/subtype_LUAD_vs_LUSC_patch.yaml --mode test \
    --feature_root /path/to/patch_embeddings
```

This loads the checkpoint of each fold and writes per-slide predictions.

### Step 4: Post-hoc OOD detection

```bash
cd MIL/abMIL
bash run_experiment.sh --config configs/subtype_LUAD_vs_LUSC_patch.yaml --mode ood --method maha \
    --feature_root /path/to/patch_embeddings
```

For each fold this loads the trained checkpoint, scores every test slide with
the chosen method, and reports **AUROC** and **FPR95** for ID (LUAD/LUSC or
Normal/Tumor) vs the OOD slides in the split. For every method a higher score
means "more in-distribution". The fold CSV must contain OOD slides (see
`--ood_subdir` in Step 1).

Some methods need parameters fitted on ID data first. `--mode ood` runs that
fit step automatically when its output file is missing in the fold directory,
and reuses the file on later runs.

| `--method` | Score                                                                         | Fit step (data)                                   | Options                                     |
| ------------ | ----------------------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------- |
| `msp`      | Max softmax probability                                                       | none                                              |                                             |
| `energy`   | `T · logsumexp(logits / T)`                                                | none                                              | `--temp` (default 1.0)                    |
| `react`    | Energy after clipping the bag embedding                                       | `get_react_threshold.py` (val, 90th percentile) | `--temp`                                  |
| `maha`     | Mahalanobis++: negative min class distance of the L2-normalised bag embedding | `fit_maha.py` (train)                           |                                             |
| `vim`      | ViM: virtual logit from the classifier hidden layer's null space + energy     | `fit_vim.py` (train)                            | `--vim_dim` (default 128)                 |
| `knn`      | Negative distance to the K-th nearest train feature (faiss)                   | `fit_knn.py` (train)                            | `--knn_K` (default 50), `--knn_use_bag` |
| `residual` | Negative norm of the null-space projection (ViM without energy)               | `fit_residual.py` (train)                       | `--residual_dim` (default 128)            |

`knn` uses the 256-d classifier hidden feature by default; `--knn_use_bag`
switches to the bag embedding (`in_dim`-d). The fit scripts and `ood_test.py` take the
same `--config`, `--fold` and `--feature_root` arguments and can be run on
their own, e.g.:

```bash
cd MIL/abMIL
python ood_methods/fit_vim.py --config configs/subtype_LUAD_vs_LUSC.yaml --fold 0 --feature_root /path/to/slide_embeddings --vim_dim 128
python ood_test.py --config configs/subtype_LUAD_vs_LUSC.yaml --fold 0 --feature_root /path/to/slide_embeddings --method vim --vim_dim 128
```

Only post-hoc methods are included. Training-time OOD methods (e.g. HaMOS) are
not part of this repo.

### Outputs

```
experiments/<experiment_name>/fold_<k>/
├── classifier.pth      # best classifier weights
├── abmil.pth           # best AB-MIL pooling weights
├── probability.csv     # --mode test: filename, label, pred, correct, prob, ood_label
├── ood_result_<method>.csv   # --mode ood: filename, ood_label, score
└── <fitted params>     # react_threshold.txt, maha_params.pt, vim_params_dim*.pt,
                        # knn_params_K*_{hidden,bag}.npz, residual_params_dim*.pt
```

`prob` is the predicted probability of class 1. Classification metrics (AUC,
accuracy, precision, recall, F1 at threshold 0.5, computed on ID slides only)
are logged to WandB under `train/`, `val/` and `test/`. OOD metrics are logged
as `ood/auroc` and `ood/fpr95`.

### Running without WandB

Call the Python entry points directly and leave out `--use_wandb`:

```bash
cd MIL/abMIL
python train.py      --config configs/detection_LUAD_patch.yaml --fold 0 --feature_root /path/to/patch_embeddings
python test.py       --config configs/detection_LUAD_patch.yaml --fold 0 --feature_root /path/to/patch_embeddings
python train_slide.py --config configs/detection_LUAD_slide.yaml --fold 0 --feature_root /path/to/slide_embeddings
python test_slide.py  --config configs/detection_LUAD_slide.yaml --fold 0 --feature_root /path/to/slide_embeddings
python ood_test.py    --config configs/detection_LUAD_slide.yaml --fold 0 --feature_root /path/to/slide_embeddings --method msp
```

Classification metrics are then not logged anywhere, but the test scripts
still write `probability.csv`. `ood_test.py` prints AUROC / FPR95 and writes
its result CSV. For OOD methods that need a fit step,
run the matching `ood_methods/fit_*.py` script first (see Step 4).

---

## Configuration

Config fields (see the files in `MIL/abMIL/configs/`):

| Field                                          | Description                                                                 |
| ---------------------------------------------- | --------------------------------------------------------------------------- |
| `experiment_name`                            | Name of the output folder under`save_base_dir`                            |
| `task_type`                                  | `patch` or `slide`; picks the entry point in `run_experiment.sh`      |
| `csv_dir`                                    | Directory with`dataset_fold_{k}.csv`                                      |
| `feature_root`                               | Root of the feature files. Machine-specific;`null` in the shipped configs |
| `in_dim`                                     | Feature dimension of the foundation model (768 for CHIEF; default 768)      |
| `batch_size`                                 | Slides per step, slide level only (default 4); patch level always uses 1 bag |
| `save_base_dir`                              | Where checkpoints and predictions are written                               |
| `wandb_project`                              | WandB project name                                                          |
| `lr`, `num_epochs`, `patience`, `seed` | Training hyperparameters                                                    |

Relative paths in `csv_dir` and `save_base_dir` resolve against the **repo
root**, not the current directory.

`feature_root` must come from either `--feature_root` on the command line or
the config. If neither is set, the scripts exit with an error. A command-line
value overrides the config, and the same holds for the other hyperparameters
(`--in_dim`, `--batch_size`, `--lr`, `--num_epochs`, `--patience`, `--seed`,
`--weight_decay`, `--class_weights`). Run `python train.py --help` for the
full list.

---

## Known limitations

- Patch-level training uses batch size 1 (one bag per step); `batch_size`
  only applies to slide-level training.
- The split scripts assume TCGA barcode file names (see Setup, step 3).

---

## License

This project is released under the [MIT License](LICENSE), Copyright (c) 2026 IIR.

## Third-party code

`MIL/abMIL/ood_methods/vim.py`, `knn.py` and `residual.py` are adapted from
[OpenMIBOOD](https://github.com/remic-othr/OpenMIBOOD) (MIT License). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the license text.
