"""Resume the prepared Colab smoke after a diagnosed setup-only failure."""
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time


def _continue_blind_smoke():
    root = Path("/content/blind_diffusion_ee1bb70_20261120")
    state_path, log_path = root / "colab_job_state.json", root / "colab_execution.log"
    previous = json.loads(state_path.read_text())
    if previous.get("status") != "failed":
        raise RuntimeError("Only resume a recorded failed workload; do not duplicate running/completed work")
    env = dict(os.environ)
    env.update(PATH=str(root / ".venv/bin") + os.pathsep + env["PATH"],
               PYTHONPATH=str(root / "src") + os.pathsep + str(root),
               MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", OMP_NUM_THREADS="2", MPLBACKEND="Agg")
    state_path.write_text(json.dumps({"stage": "two_stage_flow_smoke", "status": "running"}) + "\n")
    started = time.monotonic()
    print("COLAB_STAGE resume_two_stage_flow_smoke", flush=True)
    with log_path.open("a") as log:
        proc = subprocess.Popen(["bash", "scripts/run_recovery_v2_smoke.sh"], cwd=root, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
        while True:
            try:
                code = proc.wait(timeout=20)
                break
            except subprocess.TimeoutExpired:
                print(f"COLAB_PROGRESS smoke elapsed={int(time.monotonic()-started)}s", flush=True)
    if code:
        state_path.write_text(json.dumps({"stage": "two_stage_flow_smoke", "status": "failed", "exit_code": code}) + "\n")
        print("\n".join(log_path.read_text(errors="replace").splitlines()[-80:]), flush=True)
        raise RuntimeError(f"smoke failed with exit code {code}")
    output = root / "outputs/recovery_v2_smoke"
    summary = {"kind": "engineering_smoke", "status": "completed", "root": str(root)}
    for stage in ("wm", "policy", "clean", "blind"):
        name = "metrics.json" if stage == "wm" else "diffusion_metrics.json"
        summary[stage] = json.loads((output / stage / name).read_text())
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


_continue_blind_smoke()
