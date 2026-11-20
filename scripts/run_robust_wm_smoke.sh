#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ROBO_DATA:-$ROOT/robomimic_datasets}"
export ROBO_DATA="$DATA_ROOT" PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"
mkdir -p "$DATA_ROOT/lift"
python scripts/cuda_preflight.py
if [[ ! -f "$DATA_ROOT/lift/low_dim_v141.hdf5" ]]; then
  python -m robomimic.scripts.download_datasets --tasks lift --dataset_types ph --hdf5_types low_dim --download_dir "$DATA_ROOT"
  cp "$DATA_ROOT/lift/ph/low_dim_v141.hdf5" "$DATA_ROOT/lift/low_dim_v141.hdf5"
fi
python -m blind_diffusion.train.train_world_model \
  exp_name=robust_wm_smoke run_dir=outputs max_steps=10 val_every=1000 num_workers=0 \
  robustness.obs_noise_std=0.10 robustness.previous_action_noise_std=0.10
