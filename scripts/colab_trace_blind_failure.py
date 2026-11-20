"""Replay the first blind failure with passive belief-error instrumentation."""
import json
import os
from pathlib import Path
import subprocess
import tarfile


def main():
    root = Path("/content/blind_diffusion_ee1bb70_20261120")
    state_path = root / "flow_trace_state.json"
    if state_path.exists():
        raise RuntimeError("Trace job already submitted; inspect its saved state")
    evaluation = json.loads((root / "flow_eval_summary.json").read_text())
    if json.loads((root / "flow_eval_state.json").read_text())["status"] != "completed":
        raise RuntimeError("Wait for the primary evaluation")
    clean = evaluation["screen"]["clean"]["episodes"][0]
    blind = evaluation["screen"]["blind"]["episodes"][0]
    if clean["seed"] != 0 or clean["success"] != 1 or blind["success"] != 0:
        raise RuntimeError("This diagnostic is specified for the observed clean-success/blind-failure seed0 pair")
    env = dict(os.environ, PYTHONPATH=str(root / "src"), MPLBACKEND="Agg", OMP_NUM_THREADS="2",
               MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    state_path.write_text(json.dumps({"status": "running", "seed": 0}) + "\n")
    paths, summary = [], {}
    try:
        for mode, duration in (("clean", 0), ("blind", 20)):
            output, log_path = root / f"flow_trace_{mode}.json", root / f"flow_trace_{mode}.log"
            paths.extend([output, log_path])
            command = [str(root / ".venv/bin/python"), "scripts/trace_diffusion_rollout.py",
                "--dataset", "robomimic_datasets/lift/low_dim_v141.hdf5",
                "--wm-checkpoint", "outputs/recovery_v2_full/wm/checkpoints/best.pt",
                "--diff-checkpoint", "outputs/recovery_v2_full/flow/checkpoints/best.pt",
                "--episodes", "1", "--max-steps", "500", "--seed", "0",
                "--blackout-duration", str(duration), "--record-reconstruction", "--output", str(output)]
            print(f"COLAB_STAGE trace_{mode}_seed0", flush=True)
            with log_path.open("w") as log:
                process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                while True:
                    try:
                        code = process.wait(timeout=20)
                        break
                    except subprocess.TimeoutExpired:
                        print(f"COLAB_PROGRESS trace_{mode}", flush=True)
            if code:
                print("\n".join(log_path.read_text(errors="replace").splitlines()[-60:]), flush=True)
                raise RuntimeError(f"Trace {mode} failed with exit code {code}")
            episode = json.loads(output.read_text())["episodes"][0]
            rows = episode["trajectory"]
            reference_episode = clean if mode == "clean" else blind
            if rows[0]["observation"] != reference_episode["initial_observation"]:
                raise RuntimeError("Trace does not match the primary starting observation")
            summary[mode] = {k: v for k, v in episode.items() if k != "trajectory"}
            summary[mode]["outcome_reproduced"] = bool(episode["success"]) == bool(reference_episode["success"])
            summary[mode]["belief_error_by_step_range"] = {}
            for start, stop in ((0, 5), (5, 20), (20, 50), (50, 100), (100, 500)):
                segment = rows[start:stop]
                if not segment:
                    continue
                summary[mode]["belief_error_by_step_range"][f"{start}:{stop}"] = {
                    "normalized_mse": sum(r["observation_mse_normalized"] for r in segment) / len(segment),
                    "by_feature": {key: sum(r["observation_mse_by_feature"][key] for r in segment) / len(segment)
                                   for key in segment[0]["observation_mse_by_feature"]},
                    "position_error_m_by_feature": {
                        key: sum(r["position_error_m_by_feature"][key] for r in segment) / len(segment)
                        for key in segment[0]["position_error_m_by_feature"]}}
        summary["interpretation"] = "Seed0 was selected after its blind failure. Passive decoder errors diagnose prediction mismatch; they do not isolate the cause of failed control or estimate population success."
        summary_path = root / "flow_trace_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        archive_path = Path("/content/blind_diffusion_flow_trace_ee1bb70.tar.gz")
        state_path.write_text(json.dumps({"status": "completed", "archive": str(archive_path)}) + "\n")
        with tarfile.open(archive_path, "w:gz") as archive:
            for path in paths + [summary_path, state_path]:
                archive.add(path, arcname=path.name)
        print("COLAB_TRACE " + json.dumps(summary), flush=True)
        print(f"COLAB_ARTIFACT {archive_path}", flush=True)
    except Exception as error:
        state_path.write_text(json.dumps({"status": "failed", "error": str(error)}) + "\n")
        raise


main()
