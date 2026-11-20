#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Vast setup creates this environment; the RunPod image uses its configured
# base Python.  Keep the experiment command identical on both backends.
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate blind_diffusion_py310
fi
export ROBO_DATA="$ROOT/robomimic_datasets" PYTHONPATH="$ROOT/src:$ROOT"
mkdir -p "$ROBO_DATA/lift" "$ROOT/outputs/robustness_screen_v1"
if [[ ! -f "$ROBO_DATA/lift/low_dim_v141.hdf5" ]]; then
 python -m robomimic.scripts.download_datasets --tasks lift --dataset_types ph --hdf5_types low_dim --download_dir "$ROBO_DATA"
 cp "$ROBO_DATA/lift/ph/low_dim_v141.hdf5" "$ROBO_DATA/lift/low_dim_v141.hdf5"
fi
python scripts/screen_wm_robustness_v1.py --dataset "$ROBO_DATA/lift/low_dim_v141.hdf5" --trace "$ROOT/research/experiments/20260825_closed_loop_trace_forensics/trace.json" --wm "$ROOT/research/experiments/20260825_offline_counterfactual_rollout/source_checkpoints/wm_best.pt" --diff "$ROOT/research/experiments/20260825_offline_counterfactual_rollout/source_checkpoints/diffusion_best.pt" --output "$ROOT/outputs/robustness_screen_v1/result.json" --steps 1000
