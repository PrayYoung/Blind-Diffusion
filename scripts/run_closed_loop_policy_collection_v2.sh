#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate blind_diffusion_py310
fi
export ROBO_DATA="$ROOT/robomimic_datasets" PYTHONPATH="$ROOT/src:$ROOT"
mkdir -p "$ROBO_DATA/lift" "$ROOT/outputs/closed_loop_policy_collection_v2"
if [[ ! -f "$ROBO_DATA/lift/low_dim_v141.hdf5" ]]; then
  python -m robomimic.scripts.download_datasets --tasks lift --dataset_types ph --hdf5_types low_dim --download_dir "$ROBO_DATA"
  cp "$ROBO_DATA/lift/ph/low_dim_v141.hdf5" "$ROBO_DATA/lift/low_dim_v141.hdf5"
fi
python scripts/trace_diffusion_rollout.py \
  --dataset "$ROBO_DATA/lift/low_dim_v141.hdf5" \
  --wm-checkpoint "$ROOT/research/experiments/20260825_offline_counterfactual_rollout/source_checkpoints/wm_best.pt" \
  --diff-checkpoint "$ROOT/research/experiments/20260825_offline_counterfactual_rollout/source_checkpoints/diffusion_best.pt" \
  --episodes 4 --max-steps 500 --seed 0 \
  --output "$ROOT/outputs/closed_loop_policy_collection_v2/offpolicy_transitions.json"
