"""Execute the held-out offset audit after the full Colab WM job finishes."""
import json
import os
from pathlib import Path
import subprocess


def _audit_blind_world_model():
    root = Path("/content/blind_diffusion_ee1bb70_20261120")
    state = json.loads((root / "wm_full_state.json").read_text())
    if state.get("status") != "completed":
        raise RuntimeError("Wait for the existing full WM job; do not overlap workloads")
    env = dict(os.environ)
    env.update(PYTHONPATH=str(root / "src"), MPLBACKEND="Agg", OMP_NUM_THREADS="2")
    output = root / "wm_prior_audit.json"
    command = [str(root / ".venv/bin/python"), "scripts/audit_prior_predictions.py",
        "--checkpoint", "outputs/recovery_v2_full/wm/checkpoints/best.pt",
        "--dataset", "robomimic_datasets/lift/low_dim_v141.hdf5", "--output", str(output)]
    process = subprocess.Popen(command, cwd=root, env=env)
    while True:
        try:
            code = process.wait(timeout=20)
            break
        except subprocess.TimeoutExpired:
            print("COLAB_PROGRESS heldout_prior_audit", flush=True)
    if code:
        raise RuntimeError(f"Prior audit failed with exit code {code}")
    result = json.loads(output.read_text())
    print("COLAB_AUDIT " + json.dumps(result), flush=True)


_audit_blind_world_model()
