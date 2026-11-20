#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT_DIR/outputs/offline_counterfactual_rollout"
export ROBO_DATA="${ROBO_DATA:-$ROOT_DIR/robomimic_datasets}"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR:${PYTHONPATH:-}"
DATASET="$ROBO_DATA/lift/low_dim_v141.hdf5"
DOWNLOADED="$ROBO_DATA/lift/ph/low_dim_v141.hdf5"
WM="$ROOT_DIR/research/experiments/20260825_offline_counterfactual_rollout/source_checkpoints/wm_best.pt"
DIFF="$ROOT_DIR/research/experiments/20260825_offline_counterfactual_rollout/source_checkpoints/diffusion_best.pt"

mkdir -p "$OUT" "$ROBO_DATA"
if [[ ! -f "$DATASET" ]]; then
  python -m robomimic.scripts.download_datasets --tasks lift --dataset_types ph --hdf5_types low_dim --download_dir "$ROBO_DATA"
fi
if [[ -f "$DOWNLOADED" && ! -f "$DATASET" ]]; then
  mkdir -p "$(dirname "$DATASET")"
  cp "$DOWNLOADED" "$DATASET"
fi
test -f "$DATASET"
test -f "$WM"
test -f "$DIFF"

python scripts/offline_counterfactual_rollout.py \
  --dataset "$DATASET" \
  --wm-checkpoint "$WM" \
  --diff-checkpoint "$DIFF" \
  --output "$OUT/result.json" \
  --demo-start 0 \
  --num-demos 0 \
  --steps 50 \
  --sampler-steps 10 \
  --seed 123
