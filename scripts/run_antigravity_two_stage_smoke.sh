#!/usr/bin/env bash
set -euo pipefail

# Cheap integration check only. It validates artifact hand-off and RHC, not
# task quality or the 20k/15-epoch scientific protocol.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="$ROOT_DIR/outputs/antigravity_two_stage_smoke"
DATA_ROOT="${ROBO_DATA:-$ROOT_DIR/robomimic_datasets}"
DATASET="$DATA_ROOT/lift/low_dim_v141.hdf5"
DOWNLOADED="$DATA_ROOT/lift/ph/low_dim_v141.hdf5"
export ROBO_DATA="$DATA_ROOT"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT"

if [[ ! -f "$DATASET" ]]; then
  python -m robomimic.scripts.download_datasets --tasks lift --dataset_types ph --hdf5_types low_dim --download_dir "$DATA_ROOT"
fi
if [[ -f "$DOWNLOADED" && ! -f "$DATASET" ]]; then
  mkdir -p "$(dirname "$DATASET")"
  cp "$DOWNLOADED" "$DATASET"
fi
test -f "$DATASET"

python scripts/train_wm.py task=lift max_steps=20 val_every=20 log_every=10 batch_size=16 num_workers=0 normalize_obs=true normalize_action=false run_dir="$OUTPUT_ROOT" exp_name=wm
WM_CHECKPOINT="$OUTPUT_ROOT/wm/checkpoints/best.pt"
test -f "$WM_CHECKPOINT"

python scripts/train_diffusion.py task=lift epochs=1 max_train_steps=3 batch_size=4 num_workers=0 normalize_obs=true normalize_action=false wm_checkpoint="$WM_CHECKPOINT" run_dir="$OUTPUT_ROOT" exp_name=diffusion
DIFF_CHECKPOINT="$OUTPUT_ROOT/diffusion/checkpoints/diffusion_smoke.pt"
test -f "$DIFF_CHECKPOINT"

python scripts/eval_diffusion.py task=lift episodes=1 max_steps=5 normalize_action=false sampler=ddim inference_steps=2 wm_checkpoint="$WM_CHECKPOINT" diff_checkpoint="$DIFF_CHECKPOINT" run_dir="$OUTPUT_ROOT" exp_name=eval

python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/antigravity_two_stage_smoke")
payload = {
    "status": "PASS",
    "wm_checkpoint": str(root / "wm/checkpoints/best.pt"),
    "diffusion_checkpoint": str(root / "diffusion/checkpoints/diffusion_smoke.pt"),
    "evaluation": json.loads((root / "eval/diffusion_metrics.json").read_text()),
    "sampler": "ddim",
    "inference_steps": 2,
    "rhc": "seq[:, :, 0]",
    "action_normalization": False,
}
(root / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
