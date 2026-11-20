#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ROBO_DATA:-$ROOT_DIR/robomimic_datasets}"
OUT="$ROOT_DIR/outputs/repaired_performance_v2_100k"
DATASET="$DATA_ROOT/lift/low_dim_v141.hdf5"
DOWNLOADED="$DATA_ROOT/lift/ph/low_dim_v141.hdf5"

export ROBO_DATA="$DATA_ROOT"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
mkdir -p "$DATA_ROOT" "$OUT"

if [[ ! -f "$DATASET" ]]; then
  python -m robomimic.scripts.download_datasets --tasks lift --dataset_types ph --hdf5_types low_dim --download_dir "$DATA_ROOT"
fi
if [[ -f "$DOWNLOADED" && ! -f "$DATASET" ]]; then
  mkdir -p "$(dirname "$DATASET")"
  cp "$DOWNLOADED" "$DATASET"
fi
test -f "$DATASET"

for seed in 0 1 2; do
  run="$OUT/seed_${seed}"
  python scripts/train_wm.py task=lift seed="$seed" max_steps=100000 val_every=500 log_every=100 batch_size=64 num_workers=2 run_dir="$run" exp_name=wm
  ckpt="$run/wm/checkpoints/best.pt"
  test -f "$ckpt"
  python scripts/eval.py eval.mode=mpc task=lift seed="$seed" checkpoint="$ckpt" episodes=20 max_steps=500 run_dir="$run" exp_name=eval
  python scripts/diagnose_repaired_control.py --checkpoint "$ckpt" --checkpoint-seed "$seed" --max-steps=500 --policies mpc --output "$run/trace_mpc.json"
done

python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/repaired_performance_v2_100k")
rows = []
for seed in (0, 1, 2):
    run = root / f"seed_{seed}"
    rows.append({
        "seed": seed,
        "checkpoint": str(run / "wm/checkpoints/best.pt"),
        "training_steps": 100000,
        "episodes": 20,
        "train": json.loads((run / "wm/metrics.json").read_text()),
        "eval": json.loads((run / "eval/metrics.json").read_text()),
        "trace": str(run / "trace_mpc.json"),
    })
aggregate = {
    "protocol": "reconstructed-performance-v2-100k",
    "seeds": [0, 1, 2],
    "episodes_per_seed": 20,
    "total_episodes": 60,
    "max_episode_steps": 500,
    "runs": rows,
}
(root / "result.json").write_text(json.dumps(aggregate, indent=2) + "\n")
print(json.dumps(aggregate, indent=2))
PY
