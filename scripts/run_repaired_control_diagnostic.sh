#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT_DIR/outputs/repaired_control_diagnostic_v1"
export ROBO_DATA="${ROBO_DATA:-$ROOT_DIR/robomimic_datasets}"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
DATASET="$ROBO_DATA/lift/low_dim_v141.hdf5"
DOWNLOADED="$ROBO_DATA/lift/ph/low_dim_v141.hdf5"
mkdir -p "$OUT" "$ROBO_DATA"

if [[ ! -f "$DATASET" ]]; then
  python -m robomimic.scripts.download_datasets --tasks lift --dataset_types ph --hdf5_types low_dim --download_dir "$ROBO_DATA"
fi
if [[ -f "$DOWNLOADED" && ! -f "$DATASET" ]]; then
  mkdir -p "$(dirname "$DATASET")"
  cp "$DOWNLOADED" "$DATASET"
fi
test -f "$DATASET"

for seed in 0 1 2; do
  python scripts/diagnose_repaired_control.py \
    --checkpoint ".runpod/checkpoint_staging/seed_${seed}.pt" \
    --checkpoint-seed "$seed" \
    --max-steps 100 \
    --output "$OUT/seed_${seed}.json"
done

python - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path("outputs/repaired_control_diagnostic_v1")
per_seed = [json.loads((root / f"seed_{seed}.json").read_text()) for seed in (0, 1, 2)]
mpc = [next(r for r in row["rollouts"] if r["policy"] == "mpc") for row in per_seed]
first_actions = np.asarray([r["trace"][0]["action_environment"] for r in mpc], dtype=float)
summary = {
    "protocol": "repaired-control-diagnostic-v1",
    "checkpoint_seeds": [0, 1, 2],
    "episodes_per_policy_per_seed": 1,
    "max_steps": 100,
    "mpc_first_action_pairwise_l2": [float(np.linalg.norm(first_actions[i] - first_actions[j]))
                                      for i, j in ((0, 1), (0, 2), (1, 2))],
    "per_seed": per_seed,
}
(root / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
PY
