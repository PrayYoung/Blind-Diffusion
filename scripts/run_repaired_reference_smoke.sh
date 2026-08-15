#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="blind_diffusion_py310"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="$ROOT_DIR/outputs/repaired_reference_smoke"
RUNS_ROOT="$OUTPUT_ROOT/runs"
EVAL_ROOT="$OUTPUT_ROOT/eval_runs"
DATA_ROOT="${ROBO_DATA:-$ROOT_DIR/robomimic_datasets}"
DATASET_PATH="$DATA_ROOT/lift/low_dim_v141.hdf5"
DOWNLOADED_PATH="$DATA_ROOT/lift/ph/low_dim_v141.hdf5"

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

export ROBO_DATA="$DATA_ROOT"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR:${PYTHONPATH:-}"

mkdir -p "$OUTPUT_ROOT" "$RUNS_ROOT" "$EVAL_ROOT" "$DATA_ROOT"

python - <<'PY'
import json
import platform
import sys

import mujoco
import robomimic
import robosuite
import torch

payload = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "mujoco": mujoco.__version__,
    "robomimic": getattr(robomimic, "__version__", "unknown"),
    "robosuite": getattr(robosuite, "__version__", "unknown"),
}
print(json.dumps(payload, indent=2))
PY

if [[ ! -f "$DATASET_PATH" ]]; then
  python -m robomimic.scripts.download_datasets \
    --tasks lift \
    --dataset_types ph \
    --hdf5_types low_dim \
    --download_dir "$DATA_ROOT"
fi

if [[ -f "$DOWNLOADED_PATH" && ! -f "$DATASET_PATH" ]]; then
  mkdir -p "$(dirname "$DATASET_PATH")"
  cp "$DOWNLOADED_PATH" "$DATASET_PATH"
fi

test -f "$DATASET_PATH"

PYTHONPATH=src python -m unittest \
  tests.test_rssm \
  tests.test_datasets \
  tests.test_metrics \
  tests.test_env_metadata

python scripts/train_wm.py \
  task=lift \
  max_steps=20 \
  val_every=20 \
  log_every=10 \
  batch_size=16 \
  num_workers=0 \
  run_dir="$RUNS_ROOT" \
  exp_name=wm_lift_smoke

CHECKPOINT_PATH="$RUNS_ROOT/wm_lift_smoke/checkpoints/best.pt"
test -f "$CHECKPOINT_PATH"

python scripts/eval.py \
  eval.mode=mpc \
  task=lift \
  checkpoint="$CHECKPOINT_PATH" \
  episodes=2 \
  max_steps=200 \
  run_dir="$EVAL_ROOT" \
  exp_name=eval_lift_smoke

python - <<'PY'
import json
import os
import platform
from pathlib import Path

import torch

root = Path("outputs/repaired_reference_smoke")
train_metrics_path = root / "runs" / "wm_lift_smoke" / "metrics.json"
eval_metrics_path = root / "eval_runs" / "eval_lift_smoke" / "metrics.json"
checkpoint_path = root / "runs" / "wm_lift_smoke" / "checkpoints" / "best.pt"

payload = {
    "status": "PASS",
    "dataset_path": os.environ["ROBO_DATA"] + "/lift/low_dim_v141.hdf5",
    "checkpoint_path": str(checkpoint_path),
    "hardware": {
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    },
    "runtime": {
        "python": platform.python_version(),
    },
    "train_metrics": json.loads(train_metrics_path.read_text()) if train_metrics_path.exists() else None,
    "eval_metrics": json.loads(eval_metrics_path.read_text()) if eval_metrics_path.exists() else None,
}
(root / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
