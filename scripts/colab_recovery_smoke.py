"""Execute via colab exec -s <existing-session> -f this_file.

Uses a task-specific directory and Python 3.10 environment. Does not modify
the notebook kernel packages, mount Drive, allocate or stop a runtime.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


root = Path("/content/blind_diffusion_ee1bb70_20261120")
payload = Path("/content/blind_diffusion_payload_ee1bb70.tar.gz")
if not payload.is_file():
    raise RuntimeError("Upload the code-only payload before executing this script")
root.mkdir(exist_ok=False)
with tarfile.open(payload, "r:gz") as archive:
    archive.extractall(root, filter="data")
log_path = root / "colab_execution.log"
state_path = root / "colab_job_state.json"
env = dict(os.environ)
env["UV_PYTHON_INSTALL_DIR"] = str(root / "interpreters")
env["UV_CACHE_DIR"] = str(root / "uv_cache")


def run_stage(name, command):
    print(f"COLAB_STAGE {name}", flush=True)
    state_path.write_text(json.dumps({"stage": name, "status": "running", "root": str(root)}) + "\n")
    started = time.monotonic()
    with log_path.open("a") as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        while True:
            try:
                code = process.wait(timeout=20)
                break
            except subprocess.TimeoutExpired:
                print(f"COLAB_PROGRESS {name} elapsed={int(time.monotonic()-started)}s", flush=True)
    if code:
        state_path.write_text(json.dumps({"stage": name, "status": "failed", "exit_code": code}) + "\n")
        print("\n".join(log_path.read_text(errors="replace").splitlines()[-60:]), flush=True)
        raise RuntimeError(f"{name} failed with exit code {code}")


uv = shutil.which("uv")
if not uv:
    run_stage("bootstrap_uv", [sys.executable, "-m", "pip", "install", "--quiet", "--target", str(root / "tools"), "uv"])
    uv = str(root / "tools/bin/uv")
run_stage("python310_environment", [uv, "venv", "--managed-python", "--python", "3.10", "--seed", str(root / ".venv")])
python = str(root / ".venv/bin/python")
run_stage("pytorch", [python, "-m", "pip", "install", "--quiet", "--index-url", "https://download.pytorch.org/whl/cu124",
                      "torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0"])
run_stage("training_dependencies", [python, "-m", "pip", "install", "--quiet", "-r", "requirements-vast.txt"])
run_stage("simulator_dependencies", [python, "-m", "pip", "install", "--quiet", "mujoco==2.3.7", "robosuite==1.4.1"])
run_stage("robomimic", [python, "-m", "pip", "install", "--quiet", "--no-deps", "robomimic==0.3.0", "-e", "."])
env["PATH"] = str(root / ".venv/bin") + os.pathsep + env["PATH"]
env["PYTHONPATH"] = str(root / "src") + os.pathsep + str(root)
env["MUJOCO_GL"] = "egl"
env["PYOPENGL_PLATFORM"] = "egl"
env["OMP_NUM_THREADS"] = "2"
env["MPLBACKEND"] = "Agg"
run_stage("two_stage_flow_smoke", ["bash", "scripts/run_recovery_v2_smoke.sh"])
output = root / "outputs/recovery_v2_smoke"
summary = {"kind": "engineering_smoke", "status": "completed", "root": str(root)}
for stage in ("wm", "policy"):
    name = "metrics.json" if stage == "wm" else "diffusion_metrics.json"
    summary[stage] = json.loads((output / stage / name).read_text())
for stage in ("clean", "blind"):
    summary[stage] = json.loads((output / stage / "diffusion_metrics.json").read_text())
summary_path = root / "smoke_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
archive_path = Path("/content/blind_diffusion_smoke_ee1bb70.tar.gz")
state_path.write_text(json.dumps({"stage": "complete", "status": "completed", "archive": str(archive_path)}) + "\n")
with tarfile.open(archive_path, "w:gz") as archive:
    archive.add(root / "outputs", arcname="outputs")
    for path in (summary_path, log_path, state_path):
        archive.add(path, arcname=path.name)
print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
print(f"COLAB_ARTIFACT {archive_path}", flush=True)
