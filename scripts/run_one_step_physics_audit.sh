#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ROBO_DATA:-$ROOT/robomimic_datasets}"
DATASET="$DATA_ROOT/lift/low_dim_v141.hdf5"
DOWNLOADED="$DATA_ROOT/lift/ph/low_dim_v141.hdf5"
OUT="$ROOT/outputs/one_step_physics_audit"
AUDIT_ARGS=()
if [[ -n "${AUDIT_DEMO_COUNT:-}" ]]; then AUDIT_ARGS+=(--demo-count "$AUDIT_DEMO_COUNT"); fi
if [[ -n "${AUDIT_ANCHORS:-}" ]]; then read -r -a AUDIT_ANCHOR_ARRAY <<< "$AUDIT_ANCHORS"; AUDIT_ARGS+=(--anchors "${AUDIT_ANCHOR_ARRAY[@]}"); fi
if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; conda activate blind_diffusion_py310; fi
export ROBO_DATA="$DATA_ROOT" PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}" MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
mkdir -p "$DATA_ROOT" "$OUT"
python scripts/cuda_preflight.py
if [[ ! -f "$DATASET" ]]; then python -m robomimic.scripts.download_datasets --tasks lift --dataset_types ph --hdf5_types low_dim --download_dir "$DATA_ROOT"; fi
if [[ -f "$DOWNLOADED" && ! -f "$DATASET" ]]; then mkdir -p "$(dirname "$DATASET")"; cp "$DOWNLOADED" "$DATASET"; fi
python scripts/audit_one_step_physics.py --dataset "$DATASET" --wm-checkpoint "$ROOT/research/experiments/20260825_offline_counterfactual_rollout/source_checkpoints/wm_best.pt" --diff-checkpoint "$ROOT/research/experiments/20260825_offline_counterfactual_rollout/source_checkpoints/diffusion_best.pt" --output "$OUT/result.json" "${AUDIT_ARGS[@]}"
