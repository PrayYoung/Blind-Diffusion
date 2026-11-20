"""Full versioned WM gate on the already prepared Colab T4; no allocation."""
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time


def _train_blind_world_model():
    root = Path("/content/blind_diffusion_ee1bb70_20261120")
    if not (root / "smoke_summary.json").is_file():
        raise RuntimeError("Verify and download the two-stage smoke before the full WM run")
    state_path, log_path = root / "wm_full_state.json", root / "wm_full.log"
    if state_path.exists():
        raise RuntimeError("Full-WM state already exists; inspect it instead of submitting a duplicate")
    env = dict(os.environ)
    env.update(PATH=str(root / ".venv/bin") + os.pathsep + env["PATH"],
               PYTHONPATH=str(root / "src") + os.pathsep + str(root),
               ROBO_DATA=str(root / "robomimic_datasets"), MPLBACKEND="Agg", OMP_NUM_THREADS="2")
    command = [str(root / ".venv/bin/python"), "-m", "blind_diffusion.train.train_world_model",
               "--config-name", "train_world_model_recovery", "run_dir=outputs/recovery_v2_full",
               "exp_name=wm", "seed=0", "num_workers=0"]
    output = root / "outputs/recovery_v2_full/wm"
    started = time.monotonic()
    print("COLAB_STAGE full_world_model steps=4000 batch=128 horizon=20", flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        state_path.write_text(json.dumps({"status": "running", "pid": process.pid, "command": command}) + "\n")
        while True:
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                progress = {"elapsed_seconds": int(time.monotonic() - started)}
                history = output / "logs.jsonl"
                if history.exists():
                    rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
                    if rows:
                        progress["step"] = rows[-1]["step"]
                    validation = [r for r in rows if "val_loss" in r]
                    if validation:
                        progress["last_validation"] = validation[-1]
                print("COLAB_PROGRESS " + json.dumps(progress), flush=True)
    if code:
        state_path.write_text(json.dumps({"status": "failed", "exit_code": code}) + "\n")
        print("\n".join(log_path.read_text(errors="replace").splitlines()[-70:]), flush=True)
        raise RuntimeError(f"World model workload failed with exit code {code}")
    metrics = json.loads((output / "metrics.json").read_text())
    history = [json.loads(line) for line in (output / "logs.jsonl").read_text().splitlines()]
    best = min((r for r in history if "val_loss" in r), key=lambda r: r["val_loss"])
    summary = {"status": "completed", "kind": "heldout_world_model_gate", "metrics": metrics,
               "best_validation": best, "seconds": time.monotonic() - started,
               "posterior_and_mean_prior_gate_pass": best["val_obs_loss"] <= .04 and best["val_imag_obs_loss"] <= .20,
               "prior_metric": "Mean masked MSE across offsets 1..20, not final-offset-20 MSE; final-offset audit still required."}
    (root / "wm_full_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    archive_path = Path("/content/blind_diffusion_wm_full_ee1bb70.tar.gz")
    state_path.write_text(json.dumps({"status": "completed", "archive": str(archive_path)}) + "\n")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(output, arcname="outputs/recovery_v2_full/wm")
        for path in (root / "wm_full_summary.json", state_path, log_path):
            archive.add(path, arcname=path.name)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    print(f"COLAB_ARTIFACT {archive_path}", flush=True)


_train_blind_world_model()
