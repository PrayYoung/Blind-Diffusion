"""Run the preregistered paired control screen on the existing Colab runtime."""
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time


def main():
    root = Path("/content/blind_diffusion_ee1bb70_20261120")
    trained = json.loads((root / "flow_full_state.json").read_text())
    if trained.get("status") != "completed":
        raise RuntimeError("Full flow training must finish before evaluation")
    state_path = root / "flow_eval_state.json"
    if state_path.exists():
        raise RuntimeError("Evaluation already submitted; inspect its saved state")
    env = dict(os.environ)
    env.update(PATH=str(root / ".venv/bin") + os.pathsep + env["PATH"],
               PYTHONPATH=str(root / "src"), ROBO_DATA=str(root / "robomimic_datasets"),
               MPLBACKEND="Agg", OMP_NUM_THREADS="2", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    state_path.write_text(json.dumps({"status": "running", "stage": "screen5"}) + "\n")
    outputs = []

    def paired_evaluation(label, count, seed):
        results = {}
        for mode, duration in (("clean", 0), ("blind", 20)):
            name = f"{label}_{mode}"
            output = root / "outputs/recovery_v2_full" / name
            outputs.append(output)
            command = [str(root / ".venv/bin/python"), "-m", "blind_diffusion.eval.evaluate_diffusion",
                "--config-name", "eval_flow_matching", "run_dir=outputs/recovery_v2_full",
                f"exp_name={name}", "wm_checkpoint=outputs/recovery_v2_full/wm/checkpoints/best.pt",
                "diff_checkpoint=outputs/recovery_v2_full/flow/checkpoints/best.pt",
                f"episodes={count}", f"seed={seed}", "max_steps=500", f"blackout_duration={duration}"]
            log_path = root / f"{name}.log"
            outputs.append(log_path)
            started = time.monotonic()
            print(f"COLAB_STAGE {name} episodes={count} seed={seed}", flush=True)
            with log_path.open("w") as log:
                process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                while True:
                    try:
                        code = process.wait(timeout=20)
                        break
                    except subprocess.TimeoutExpired:
                        print(f"COLAB_PROGRESS {name} elapsed={int(time.monotonic() - started)}", flush=True)
            if code:
                print("\n".join(log_path.read_text(errors="replace").splitlines()[-60:]), flush=True)
                raise RuntimeError(f"{name} failed with exit code {code}")
            results[mode] = json.loads((output / "diffusion_metrics.json").read_text())
            print("COLAB_RESULT " + json.dumps({"stage": name, "episodes": [
                {k: ep[k] for k in ("seed", "success", "return", "length")}
                for ep in results[mode]["episodes"]]}), flush=True)
        clean, blind = results["clean"]["episodes"], results["blind"]["episodes"]
        if len(clean) != count or len(blind) != count:
            raise RuntimeError("Incomplete episode records")
        if any(a["seed"] != b["seed"] or a["initial_observation"] != b["initial_observation"]
               for a, b in zip(clean, blind)):
            raise RuntimeError("Paired initial observations differ; control comparison is invalid")
        return results

    try:
        screen = paired_evaluation("screen5", 5, 0)
        successes = sum(ep["success"] for ep in screen["clean"]["episodes"])
        summary = {"screen": screen, "paired_initial_observations_equal": True,
                   "expanded": successes >= 3, "screen_clean_successes": successes}
        if successes >= 3:
            # Continue the same seed range; retain the screen's seeds 0..4.
            remaining = paired_evaluation("remaining15", 15, 5)
            summary["remaining"] = remaining
            summary["combined20"] = {mode: {
                "successes": sum(ep["success"] for group in (screen, remaining)
                                 for ep in group[mode]["episodes"]), "episodes": 20}
                for mode in ("clean", "blind")}
        summary_path = root / "flow_eval_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        archive_path = Path("/content/blind_diffusion_flow_eval_ee1bb70.tar.gz")
        state_path.write_text(json.dumps({"status": "completed", "archive": str(archive_path)}) + "\n")
        with tarfile.open(archive_path, "w:gz") as archive:
            for path in outputs + [summary_path, state_path]:
                archive.add(path, arcname=str(path.relative_to(root)))
        print("COLAB_SUMMARY " + json.dumps(summary), flush=True)
        print(f"COLAB_ARTIFACT {archive_path}", flush=True)
    except Exception as error:
        state_path.write_text(json.dumps({"status": "failed", "error": str(error)}) + "\n")
        raise


main()
