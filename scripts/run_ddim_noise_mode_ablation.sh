#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ROBO_DATA:-$ROOT/robomimic_datasets}"
DATASET="$DATA_ROOT/lift/low_dim_v141.hdf5"
DOWNLOADED="$DATA_ROOT/lift/ph/low_dim_v141.hdf5"
OUT="$ROOT/outputs/ddim_noise_mode_ablation"
WM="$ROOT/research/experiments/20260825_offline_counterfactual_rollout/source_checkpoints/wm_best.pt"
DIFF="$ROOT/research/experiments/20260825_offline_counterfactual_rollout/source_checkpoints/diffusion_best.pt"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate blind_diffusion_py310
fi

export ROBO_DATA="$DATA_ROOT"
export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
mkdir -p "$DATA_ROOT" "$OUT"

python scripts/cuda_preflight.py

if [[ ! -f "$DATASET" ]]; then
  python -m robomimic.scripts.download_datasets --tasks lift --dataset_types ph --hdf5_types low_dim --download_dir "$DATA_ROOT"
fi
if [[ -f "$DOWNLOADED" && ! -f "$DATASET" ]]; then
  mkdir -p "$(dirname "$DATASET")"
  cp "$DOWNLOADED" "$DATASET"
fi
test -f "$DATASET"
test -f "$WM"
test -f "$DIFF"

for mode in fresh fixed; do
  python scripts/trace_diffusion_rollout.py \
    --dataset "$DATASET" \
    --wm-checkpoint "$WM" \
    --diff-checkpoint "$DIFF" \
    --episodes 4 --max-steps 500 --seed 0 \
    --ddim-noise-mode "$mode" \
    --output "$OUT/${mode}.json"
done

python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/ddim_noise_mode_ablation")
result = {}
for mode in ("fresh", "fixed"):
    payload = json.loads((root / f"{mode}.json").read_text())
    episodes = payload["episodes"]
    result[mode] = {
        "successes": sum(bool(ep["success"]) for ep in episodes),
        "returns": [ep["return"] for ep in episodes],
        "gripper_sign_changes": [ep["summary"]["gripper_sign_changes"] for ep in episodes],
        "action_std": [ep["summary"]["action_std"] for ep in episodes],
        "distance_min": [ep["summary"]["distance_min"] for ep in episodes],
        "object_displacement": [ep["summary"]["object_displacement"] for ep in episodes],
        "z_norm_start_end": [ep["summary"]["z_norm_start_end"] for ep in episodes],
    }
(root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
