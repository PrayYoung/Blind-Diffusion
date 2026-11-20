#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate blind_diffusion_py310; fi
export ROBO_DATA="$ROOT/robomimic_datasets" PYTHONPATH="$ROOT/src:$ROOT"
mkdir -p "$ROBO_DATA/lift"
if [[ ! -f "$ROBO_DATA/lift/low_dim_v141.hdf5" ]]; then
  python -m robomimic.scripts.download_datasets --tasks lift --dataset_types ph --hdf5_types low_dim --download_dir "$ROBO_DATA"
  cp "$ROBO_DATA/lift/ph/low_dim_v141.hdf5" "$ROBO_DATA/lift/low_dim_v141.hdf5"
fi
python scripts/screen_policy_induced_wm_v2.py --dataset "$ROBO_DATA/lift/low_dim_v141.hdf5" --offpolicy "$ROOT/research/experiments/20260827_policy_induced_wm_v2/offpolicy_transitions_runpod_20260829T184705Z.json" --wm "$ROOT/research/experiments/20260825_offline_counterfactual_rollout/source_checkpoints/wm_best.pt" --diff "$ROOT/research/experiments/20260825_offline_counterfactual_rollout/source_checkpoints/diffusion_best.pt" --trace "$ROOT/research/experiments/20260825_closed_loop_trace_forensics/trace.json" --output "$ROOT/outputs/policy_induced_wm_v2/result.json" --steps 250
