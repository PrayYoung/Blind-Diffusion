"""Run full flow training only after both world-model gates pass."""
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time


def _train_blind_flow_policy():
    root = Path("/content/blind_diffusion_ee1bb70_20261120")
    wm = json.loads((root / "wm_full_summary.json").read_text())
    audit = json.loads((root / "wm_prior_audit.json").read_text())
    if not wm["posterior_and_mean_prior_gate_pass"] or not audit["final20_gate_pass"]:
        raise RuntimeError("WM accuracy gate failed; do not train a full blind policy")
    state_path, log_path = root / "flow_full_state.json", root / "flow_full.log"
    if state_path.exists():
        raise RuntimeError("Flow job state already exists; inspect instead of duplicating work")
    env = dict(os.environ)
    env.update(PATH=str(root / ".venv/bin") + os.pathsep + env["PATH"],
               PYTHONPATH=str(root / "src"), ROBO_DATA=str(root / "robomimic_datasets"),
               MPLBACKEND="Agg", OMP_NUM_THREADS="2")
    command = [str(root / ".venv/bin/python"), "-m", "blind_diffusion.train.train_diffusion",
        "--config-name", "train_flow_matching_recovery", "run_dir=outputs/recovery_v2_full",
        "exp_name=flow", "wm_checkpoint=outputs/recovery_v2_full/wm/checkpoints/best.pt", "seed=0", "num_workers=0"]
    output = root / "outputs/recovery_v2_full/flow"
    started = time.monotonic()
    print("COLAB_STAGE full_flow_policy epochs=100 batch=128 prior_aug=0.5", flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        state_path.write_text(json.dumps({"status": "running", "pid": process.pid, "command": command}) + "\n")
        while True:
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                progress = {"elapsed_seconds": int(time.monotonic() - started)}
                history = output / "diffusion_logs.jsonl"
                if history.exists():
                    rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
                    if rows:
                        progress["latest"] = rows[-1]
                print("COLAB_PROGRESS " + json.dumps(progress), flush=True)
    if code:
        state_path.write_text(json.dumps({"status": "failed", "exit_code": code}) + "\n")
        print("\n".join(log_path.read_text(errors="replace").splitlines()[-70:]), flush=True)
        raise RuntimeError(f"Flow training failed with exit code {code}")
    metrics = json.loads((output / "diffusion_metrics.json").read_text())
    summary = {"status": "completed", "kind": "full_flow_policy_training", "metrics": metrics,
               "seconds": time.monotonic() - started, "wm_final20_mse": audit["by_offset"]["20"]["mse"]}
    summary_path = root / "flow_full_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    archive_path = Path("/content/blind_diffusion_flow_full_ee1bb70.tar.gz")
    state_path.write_text(json.dumps({"status": "completed", "archive": str(archive_path)}) + "\n")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(output, arcname="outputs/recovery_v2_full/flow")
        for path in (summary_path, state_path, log_path):
            archive.add(path, arcname=path.name)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print(f"COLAB_ARTIFACT {archive_path}", flush=True)


_train_blind_flow_policy()
