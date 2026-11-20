"""Audit the trained policy without changing its control-screen protocol."""
import json
import os
from pathlib import Path
import subprocess


def main():
    root = Path("/content/blind_diffusion_ee1bb70_20261120")
    if json.loads((root / "flow_full_state.json").read_text()).get("status") != "completed":
        raise RuntimeError("Wait for full flow training")
    env = dict(os.environ, PYTHONPATH=str(root / "src"), MPLBACKEND="Agg", OMP_NUM_THREADS="2")
    output = root / "flow_imitation_audit.json"
    command = [str(root / ".venv/bin/python"), "scripts/audit_flow_imitation.py",
        "--checkpoint", "outputs/recovery_v2_full/flow/checkpoints/best.pt",
        "--world-model", "outputs/recovery_v2_full/wm/checkpoints/best.pt",
        "--dataset", "robomimic_datasets/lift/low_dim_v141.hdf5", "--output", str(output)]
    process = subprocess.Popen(command, cwd=root, env=env)
    while True:
        try:
            code = process.wait(timeout=20)
            break
        except subprocess.TimeoutExpired:
            print("COLAB_PROGRESS heldout_flow_imitation_audit", flush=True)
    if code:
        raise RuntimeError(f"Flow imitation audit failed with exit code {code}")
    print("COLAB_AUDIT " + output.read_text(), flush=True)


main()
