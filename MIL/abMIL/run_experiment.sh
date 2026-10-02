#!/bin/bash
# =============================================================
# run_experiment.sh  --  Unified experiment launcher
#
# Usage:
#   bash run_experiment.sh --config configs/subtype_LUAD_vs_LUSC.yaml --mode train --feature_root /path/to/slide_embeddings
#   bash run_experiment.sh --config configs/subtype_LUAD_vs_LUSC.yaml --mode test  --feature_root /path/to/slide_embeddings
#   bash run_experiment.sh --config configs/detection_LUAD_patch.yaml --mode train --feature_root /path/to/patch_embeddings --folds "0 1"
#
# Post-hoc OOD evaluation (fits the method's parameters first if they are missing):
#   bash run_experiment.sh --config <cfg> --mode ood --method msp       --feature_root <root>
#   bash run_experiment.sh --config <cfg> --mode ood --method energy    --feature_root <root> [--temp 1.0]
#   bash run_experiment.sh --config <cfg> --mode ood --method react     --feature_root <root>
#   bash run_experiment.sh --config <cfg> --mode ood --method maha      --feature_root <root>
#   bash run_experiment.sh --config <cfg> --mode ood --method vim       --feature_root <root> [--vim_dim 128]
#   bash run_experiment.sh --config <cfg> --mode ood --method knn       --feature_root <root> [--knn_K 50] [--knn_use_bag]
#   bash run_experiment.sh --config <cfg> --mode ood --method residual  --feature_root <root> [--residual_dim 128]
#
# --feature_root may be omitted if the config sets feature_root.
# =============================================================
set -e

CONFIG=""
MODE="train"
FOLDS="0 1 2 3"
FEATURE_ROOT=""
METHOD="msp"
TEMP="1.0"
VIM_DIM="128"
KNN_K="50"
KNN_USE_BAG_FIT=""      # flag for fit_knn.py:  --use_bag_embedding
KNN_USE_BAG_TEST=""     # flag for ood_test.py: --knn_use_bag
RESIDUAL_DIM="128"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)       CONFIG="$2";   shift 2 ;;
        --mode)         MODE="$2";     shift 2 ;;
        --folds)        FOLDS="$2";    shift 2 ;;
        --feature_root) FEATURE_ROOT="$2"; shift 2 ;;
        --method)       METHOD="$2";   shift 2 ;;
        --temp)         TEMP="$2";     shift 2 ;;
        --vim_dim)      VIM_DIM="$2";  shift 2 ;;
        --knn_K)        KNN_K="$2";    shift 2 ;;
        --knn_use_bag)  KNN_USE_BAG_FIT="--use_bag_embedding"; KNN_USE_BAG_TEST="--knn_use_bag"; shift ;;
        --residual_dim) RESIDUAL_DIM="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "ERROR: --config is required."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXTRA_ARGS=()
[[ -n "$FEATURE_ROOT" ]] && EXTRA_ARGS+=(--feature_root "$FEATURE_ROOT")

# Detect task_type from YAML (default: patch)
TASK_TYPE=$(python3 -c "
import yaml, sys
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)
print(c.get('task_type', 'patch'))
" "$CONFIG")

# Helper: model dir of a fold (relative save_base_dir resolves against the repo root)
get_model_dir() {
    python3 -c "
import yaml, os, sys
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)
base = c['save_base_dir']
if not os.path.isabs(base):
    base = os.path.join(sys.argv[3], base)
print(os.path.join(base, c['experiment_name'], 'fold_' + sys.argv[2]))
" "$CONFIG" "$1" "$REPO_ROOT"
}

echo "================================================"
echo " Config    : $CONFIG"
echo " Mode      : $MODE"
echo " Task type : $TASK_TYPE"
echo " Folds     : $FOLDS"
[[ -n "$FEATURE_ROOT" ]] && echo " Features  : $FEATURE_ROOT"
[[ "$MODE" == "ood" ]] && echo " Method    : $METHOD"
echo "================================================"

for FOLD in $FOLDS; do
    echo ""
    echo "--- Fold $FOLD ---"

    if [[ "$MODE" == "train" ]]; then
        if [[ "$TASK_TYPE" == "slide" ]]; then
            python "$SCRIPT_DIR/train_slide.py" --config "$CONFIG" --fold "$FOLD" "${EXTRA_ARGS[@]}" --use_wandb
        else
            python "$SCRIPT_DIR/train.py" --config "$CONFIG" --fold "$FOLD" "${EXTRA_ARGS[@]}" --use_wandb
        fi

    elif [[ "$MODE" == "test" ]]; then
        if [[ "$TASK_TYPE" == "slide" ]]; then
            python "$SCRIPT_DIR/test_slide.py" --config "$CONFIG" --fold "$FOLD" "${EXTRA_ARGS[@]}" --use_wandb
        else
            python "$SCRIPT_DIR/test.py" --config "$CONFIG" --fold "$FOLD" "${EXTRA_ARGS[@]}" --use_wandb
        fi

    elif [[ "$MODE" == "ood" ]]; then
        MODEL_DIR=$(get_model_dir "$FOLD")
        COMMON=(--config "$CONFIG" --fold "$FOLD" "${EXTRA_ARGS[@]}")
        METHOD_ARGS=()

        # ── Fit the method's parameters if missing ──────────────────────────
        case "$METHOD" in
            msp|energy) ;;
            react)
                if [[ ! -f "$MODEL_DIR/react_threshold.txt" ]]; then
                    echo "Calculating ReAct threshold for fold $FOLD..."
                    python "$SCRIPT_DIR/ood_methods/get_react_threshold.py" "${COMMON[@]}"
                fi ;;
            maha)
                if [[ ! -f "$MODEL_DIR/maha_params.pt" ]]; then
                    echo "Fitting Mahalanobis++ parameters for fold $FOLD..."
                    python "$SCRIPT_DIR/ood_methods/fit_maha.py" "${COMMON[@]}"
                fi ;;
            vim)
                METHOD_ARGS=(--vim_dim "$VIM_DIM")
                if [[ ! -f "$MODEL_DIR/vim_params_dim${VIM_DIM}.pt" ]]; then
                    echo "Fitting VIM parameters for fold $FOLD (dim=$VIM_DIM)..."
                    python "$SCRIPT_DIR/ood_methods/fit_vim.py" "${COMMON[@]}" --vim_dim "$VIM_DIM"
                fi ;;
            knn)
                FEAT_TAG=$([[ -n "$KNN_USE_BAG_FIT" ]] && echo "bag" || echo "hidden")
                METHOD_ARGS=(--knn_K "$KNN_K" $KNN_USE_BAG_TEST)
                if [[ ! -f "$MODEL_DIR/knn_params_K${KNN_K}_${FEAT_TAG}.npz" ]]; then
                    echo "Fitting KNN index for fold $FOLD (K=$KNN_K, feat=$FEAT_TAG)..."
                    python "$SCRIPT_DIR/ood_methods/fit_knn.py" "${COMMON[@]}" --K "$KNN_K" $KNN_USE_BAG_FIT
                fi ;;
            residual)
                METHOD_ARGS=(--residual_dim "$RESIDUAL_DIM")
                if [[ ! -f "$MODEL_DIR/residual_params_dim${RESIDUAL_DIM}.pt" ]]; then
                    echo "Fitting Residual parameters for fold $FOLD (dim=$RESIDUAL_DIM)..."
                    python "$SCRIPT_DIR/ood_methods/fit_residual.py" "${COMMON[@]}" --dim "$RESIDUAL_DIM"
                fi ;;
            *)
                echo "ERROR: Unknown method '$METHOD'."
                exit 1 ;;
        esac

        python "$SCRIPT_DIR/ood_test.py" "${COMMON[@]}" \
            --method "$METHOD" --temperature "$TEMP" "${METHOD_ARGS[@]}" --use_wandb

    else
        echo "ERROR: Unknown mode '$MODE'. Use train, test or ood."
        exit 1
    fi
done

echo ""
echo "All folds completed."
