#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate blind_diffusion_py310
fi
export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"
export ROBO_DATA="${ROBO_DATA:-$ROOT/robomimic_datasets}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
mkdir -p "$ROBO_DATA/lift"
python scripts/cuda_preflight.py
if [[ ! -f "$ROBO_DATA/lift/low_dim_v141.hdf5" ]]; then
  python -m robomimic.scripts.download_datasets --tasks lift --dataset_types ph --hdf5_types low_dim --download_dir "$ROBO_DATA"
  cp "$ROBO_DATA/lift/ph/low_dim_v141.hdf5" "$ROBO_DATA/lift/low_dim_v141.hdf5"
fi
RUN_ROOT=outputs/recovery_v2_smoke
python -m blind_diffusion.train.train_world_model --config-name train_world_model_recovery \
  run_dir="$RUN_ROOT" exp_name=wm max_steps=20 val_every=10 batch_size=16 num_workers=0 log_every=5
python -m blind_diffusion.train.train_diffusion --config-name train_flow_matching_recovery \
  run_dir="$RUN_ROOT" exp_name=policy wm_checkpoint="$RUN_ROOT/wm/checkpoints/best.pt" \
  max_train_steps=10 epochs=1 batch_size=16 num_workers=0 val_mc_samples=1 log_every=5
python -m blind_diffusion.eval.evaluate_diffusion --config-name eval_flow_matching \
  run_dir="$RUN_ROOT" exp_name=clean wm_checkpoint="$RUN_ROOT/wm/checkpoints/best.pt" \
  diff_checkpoint="$RUN_ROOT/policy/checkpoints/best.pt" episodes=1 max_steps=60
python -m blind_diffusion.eval.evaluate_diffusion --config-name eval_flow_matching \
  run_dir="$RUN_ROOT" exp_name=blind wm_checkpoint="$RUN_ROOT/wm/checkpoints/best.pt" \
  diff_checkpoint="$RUN_ROOT/policy/checkpoints/best.pt" episodes=1 max_steps=60 blackout_duration=20
