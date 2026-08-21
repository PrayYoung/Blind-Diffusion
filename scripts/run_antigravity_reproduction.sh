#!/usr/bin/env bash
set -euo pipefail

# Canonical recovered low-dimensional protocol. It retains Antigravity's
# training/evaluation budget but deliberately uses causal bounded direct-x0
# diffusion after the epsilon/cosine correctness diagnosis. Do not use it for
# a smoke or multi-seed sweep.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ROBO_DATA:-$ROOT_DIR/robomimic_datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/antigravity_reproduction}"
DATASET="$DATA_ROOT/lift/low_dim_v141.hdf5"
DOWNLOADED="$DATA_ROOT/lift/ph/low_dim_v141.hdf5"
SEED="${SEED:-0}"

export ROBO_DATA="$DATA_ROOT"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
mkdir -p "$DATA_ROOT" "$OUTPUT_ROOT"

if [[ ! -f "$DATASET" ]]; then
  python -m robomimic.scripts.download_datasets --tasks lift --dataset_types ph --hdf5_types low_dim --download_dir "$DATA_ROOT"
fi
if [[ -f "$DOWNLOADED" && ! -f "$DATASET" ]]; then
  mkdir -p "$(dirname "$DATASET")"
  cp "$DOWNLOADED" "$DATASET"
fi
test -f "$DATASET"

python scripts/train_wm.py task=lift seed="$SEED" max_steps=20000 seq_len=50 batch_size=64 lr=3e-4 burn_in=5 kl_free_bits=1.0 kl_scale=1.0 grad_clip=100 normalize_obs=true normalize_action=false run_dir="$OUTPUT_ROOT" exp_name=wm
WM_CHECKPOINT="$OUTPUT_ROOT/wm/checkpoints/best.pt"
test -f "$WM_CHECKPOINT"

python scripts/train_diffusion.py task=lift seed="$SEED" epochs=15 batch_size=64 lr=1e-4 normalize_obs=true normalize_action=false wm_checkpoint="$WM_CHECKPOINT" run_dir="$OUTPUT_ROOT" exp_name=diffusion
DIFF_CHECKPOINT="$OUTPUT_ROOT/diffusion/checkpoints/best.pt"
test -f "$DIFF_CHECKPOINT"

python scripts/eval_diffusion.py task=lift seed="$SEED" episodes=20 max_steps=500 normalize_action=false sampler=ddim inference_steps=10 wm_checkpoint="$WM_CHECKPOINT" diff_checkpoint="$DIFF_CHECKPOINT" run_dir="$OUTPUT_ROOT" exp_name=eval_diffusion

python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/antigravity_reproduction")
payload = {
    "protocol": "causal-bounded-x0-single-seed-v1",
    "wm_checkpoint": str(root / "wm/checkpoints/best.pt"),
    "diffusion_checkpoint": str(root / "diffusion/checkpoints/best.pt"),
    "wm": json.loads((root / "wm/metrics.json").read_text()),
    "diffusion": json.loads((root / "diffusion/diffusion_metrics.json").read_text()),
    "evaluation": json.loads((root / "eval_diffusion/diffusion_metrics.json").read_text()),
    "sampler": "ddim",
    "inference_steps": 10,
    "rhc": "seq[:, :, 0]",
    "normalize_action": False,
    "prediction_type": "x0",
}
(root / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
