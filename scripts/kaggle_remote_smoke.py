#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_TEMPLATE_DIR = REPO_ROOT / "kaggle" / "remote_smoke"
DEFAULT_BUILD_DIR = REPO_ROOT / "build" / "kaggle_remote_smoke"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "kaggle_remote_smoke_output"
DEFAULT_SLUG = "blind-diffusion-remote-smoke"
DEFAULT_TITLE = "Blind Diffusion Remote Smoke"
DEFAULT_USERNAME = "PrayYoung"
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
DEFAULT_TIMEOUT_SECONDS = 60 * 60
DEFAULT_POLL_SECONDS = 30
LOCAL_KAGGLE_CONFIG_DIR = REPO_ROOT / "build" / "kaggle_config"
STATE_ROOT = REPO_ROOT / ".codex" / "state" / "kaggle_jobs"
STATE_INDEX_PATH = STATE_ROOT / "index.json"
STATE_OUTPUTS_ROOT = STATE_ROOT / "outputs"
STATE_ANALYSIS_ROOT = STATE_ROOT / "analysis"
STATE_BUILD_ROOT = STATE_ROOT / "builds"
MAX_SLUG_LENGTH = 49
EXCLUDED_BUNDLE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "build",
    "runs",
}
EXCLUDED_BUNDLE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".png",
    ".jpg",
    ".jpeg",
    ".mp4",
    ".zip",
    ".pt",
    ".hdf5",
}


def run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=capture_output,
        env=env,
        check=False,
    )


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def require_kaggle_cli() -> None:
    if shutil.which("kaggle"):
        return
    raise SystemExit(
        "Kaggle CLI is not installed. Install it with `python3 -m pip install kaggle` "
        "and configure KAGGLE_USERNAME / KAGGLE_KEY or ~/.kaggle/kaggle.json."
    )


def resolve_username(value: str | None) -> str:
    if value:
        return value
    return os.environ.get("KAGGLE_USERNAME", DEFAULT_USERNAME)


def kernel_ref(username: str, slug: str) -> str:
    return f"{username}/{slug}"


def kaggle_env(username: str) -> dict[str, str]:
    env = os.environ.copy()
    config_dir = LOCAL_KAGGLE_CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    env["KAGGLE_CONFIG_DIR"] = str(config_dir)
    env.setdefault("KAGGLE_USERNAME", username)

    api_token = env.get("KAGGLE_API_TOKEN")
    key = env.get("KAGGLE_KEY")
    config_path = config_dir / "kaggle.json"
    if not api_token and key:
        env["KAGGLE_API_TOKEN"] = key
        api_token = key
    if key:
        config_path.write_text(json.dumps({"username": env["KAGGLE_USERNAME"], "key": key}) + "\n")
        config_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    token_path = config_dir / "access_token"
    if api_token:
        token_path.write_text(api_token.strip() + "\n")
        token_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    elif not config_path.exists():
        raise SystemExit(
            "Kaggle credentials are not available in this shell. "
            "Set KAGGLE_API_TOKEN or KAGGLE_KEY and optionally KAGGLE_USERNAME, or place kaggle.json/access_token under "
            f"{config_dir}."
        )
    return env


def build_bundle_constant() -> str:
    files: dict[str, str] = {}
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT)
        if any(part in EXCLUDED_BUNDLE_DIRS for part in rel.parts):
            continue
        if path.suffix.lower() in EXCLUDED_BUNDLE_SUFFIXES:
            continue
        try:
            files[str(rel)] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    payload = json.dumps(files, sort_keys=True).encode("utf-8")
    return base64.b64encode(gzip.compress(payload)).decode("ascii")


def build_kernel_tree(build_dir: Path, username: str, slug: str, title: str, private: bool, internet: bool) -> Path:
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    for relative_path in [
        Path("requirements-kaggle.txt"),
        Path("pyproject.toml"),
    ]:
        source = REPO_ROOT / relative_path
        target = build_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    template = (KAGGLE_TEMPLATE_DIR / "remote_smoke.py").read_text()
    bundle = build_bundle_constant()
    if "__BUNDLED_REPO_GZ_B64__" not in template:
        raise RuntimeError("remote_smoke.py template is missing bundle placeholder.")
    (build_dir / "remote_smoke.py").write_text(template.replace("__BUNDLED_REPO_GZ_B64__", bundle))

    metadata = {
        "id": kernel_ref(username, slug),
        "title": title,
        "code_file": "remote_smoke.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": private,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": internet,
        "keywords": ["benchmark", "gpu", "smoke"],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    (build_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return build_dir


def parse_status_text(stdout: str) -> str:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in lines:
        lowered = line.lower()
        if lowered.startswith("status:"):
            return line.split(":", 1)[1].strip()
    if lines:
        return lines[-1]
    return "unknown"


def git_commit_short() -> str:
    result = run_command(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Unable to determine git commit.\nstderr:\n{result.stderr}")
    return result.stdout.strip()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "experiment"


def make_experiment_slug(experiment_id: str, git_commit: str, timestamp: str) -> str:
    prefix = "bd"
    exp = slugify(experiment_id)
    ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).strftime("%Y%m%d%H%M%S")
    suffix = f"{git_commit}-{ts}"
    max_exp_len = MAX_SLUG_LENGTH - len(prefix) - len(suffix) - 2
    exp = exp[:max_exp_len].rstrip("-") or "exp"
    slug = f"{prefix}-{exp}-{suffix}"
    return slug[:MAX_SLUG_LENGTH].rstrip("-")


def state_path(experiment_id: str) -> Path:
    return STATE_ROOT / f"{slugify(experiment_id)}.json"


def ensure_state_dirs() -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_BUILD_ROOT.mkdir(parents=True, exist_ok=True)


def load_state(experiment_id: str) -> dict[str, object]:
    path = state_path(experiment_id)
    if not path.exists():
        raise FileNotFoundError(f"No saved experiment state for {experiment_id} at {path}")
    return json.loads(path.read_text())


def write_state(state: dict[str, object]) -> Path:
    ensure_state_dirs()
    path = state_path(str(state["experiment_id"]))
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    rebuild_state_index()
    return path


def rebuild_state_index() -> None:
    ensure_state_dirs()
    jobs = []
    for path in sorted(STATE_ROOT.glob("*.json")):
        if path.name == STATE_INDEX_PATH.name:
            continue
        job = json.loads(path.read_text())
        jobs.append(
            {
                "experiment_id": job["experiment_id"],
                "kaggle_slug": job["kaggle_slug"],
                "git_commit": job["git_commit"],
                "state": job["state"],
                "submitted_at": job.get("submitted_at"),
                "last_checked_at": job.get("last_checked_at"),
                "state_file": str(path.relative_to(REPO_ROOT)),
            }
        )
    STATE_INDEX_PATH.write_text(json.dumps({"jobs": jobs, "updated_at": utcnow_iso()}, indent=2, sort_keys=True) + "\n")


def status_to_job_state(status_text: str) -> str:
    lowered = status_text.lower()
    if any(token in lowered for token in ["complete", "success"]):
        return "COMPLETED"
    if any(token in lowered for token in ["failed", "error", "cancel"]):
        return "FAILED"
    if any(token in lowered for token in ["running", "queued", "starting"]):
        return "RUNNING_REMOTE"
    return "SUBMITTED"


def query_remote_status(username: str, slug: str) -> tuple[str, str, int]:
    env = kaggle_env(username)
    result = run_command(
        ["kaggle", "kernels", "status", kernel_ref(username, slug)],
        capture_output=True,
        env=env,
    )
    output = (result.stdout or "") + (result.stderr or "")
    return parse_status_text(output), output, result.returncode


def sync_remote_state(state: dict[str, object]) -> dict[str, object]:
    require_kaggle_cli()
    username = str(state["username"])
    slug = str(state["kaggle_slug"])
    status_text, output, returncode = query_remote_status(username, slug)
    state["last_checked_at"] = utcnow_iso()
    state["remote_status_text"] = status_text
    state["remote_status_output"] = output
    state["remote_status_returncode"] = returncode
    prior_state = str(state.get("state", "CREATED"))
    mapped_state = status_to_job_state(status_text)
    if prior_state not in {"RESULTS_DOWNLOADED", "ANALYZED"} or mapped_state == "FAILED":
        state["state"] = mapped_state
    write_state(state)
    return state


def default_output_dir_for_experiment(experiment_id: str) -> Path:
    return STATE_OUTPUTS_ROOT / slugify(experiment_id)


def default_analysis_dir_for_experiment(experiment_id: str) -> Path:
    return STATE_ANALYSIS_ROOT / slugify(experiment_id)


def default_build_dir_for_experiment(experiment_id: str) -> Path:
    return STATE_BUILD_ROOT / slugify(experiment_id)


def load_result_json(output_dir: Path) -> dict:
    candidates = [
        output_dir / "result.json",
        output_dir / "output" / "result.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text())
    raise FileNotFoundError(f"Could not find result.json under {output_dir}")


def find_sweep_summary(output_dir: Path) -> Path:
    preferred = []
    for path in output_dir.rglob("summary.json"):
        parts = path.parts
        if len(parts) >= 3 and parts[-3:] == ("results", "text_only_budget_sweep", "summary.json"):
            preferred.append(path)
    if not preferred:
        raise FileNotFoundError(
            "Could not find results/text_only_budget_sweep/summary.json under "
            f"{output_dir}"
        )
    return sorted(preferred)[0]


def classify_curve(summary_rows: list[dict[str, object]]) -> str:
    actual_bytes = [float(row["actual_stored_bytes_mean"]) for row in summary_rows]
    if not actual_bytes or len(set(actual_bytes)) == 1:
        return "saturates"
    if len(set(actual_bytes)) <= max(2, len(actual_bytes) // 2):
        return "coarse_step_function"
    return "approximately_continuous"


def analyze_sweep_outputs(output_dir: Path, analysis_dir: Path) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_path = find_sweep_summary(output_dir)
    summary = json.loads(summary_path.read_text())
    rows = sorted(summary["budgets"], key=lambda item: int(item["nominal_budget_bytes"]))

    nominal = [int(row["nominal_budget_bytes"]) for row in rows]
    actual = [float(row["actual_stored_bytes_mean"]) for row in rows]
    utilization = [float(row["actual_byte_utilization_ratio"]) for row in rows]
    semantic = [float(row["semantic_similarity_mean"]) for row in rows]
    qa = [float(row["scene_qa_accuracy_mean"]) for row in rows]
    dino = [float(row["dino_similarity_mean"]) for row in rows]

    plt.figure(figsize=(7, 4.5))
    plt.plot(nominal, actual, marker="o", label="actual stored bytes")
    plt.xlabel("Nominal budget bytes")
    plt.ylabel("Actual stored bytes mean")
    plt.title("Nominal Budget vs Actual Stored Bytes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    budget_plot = analysis_dir / "nominal_budget_vs_actual_bytes.png"
    plt.tight_layout()
    plt.savefig(budget_plot, dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(nominal, utilization, marker="o", color="tab:green")
    plt.xlabel("Nominal budget bytes")
    plt.ylabel("Budget utilization ratio")
    plt.title("Nominal Budget vs Utilization Ratio")
    plt.grid(True, alpha=0.3)
    utilization_plot = analysis_dir / "nominal_budget_vs_utilization_ratio.png"
    plt.tight_layout()
    plt.savefig(utilization_plot, dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [
        ("Semantic similarity", semantic),
        ("Scene QA accuracy", qa),
        ("DINO similarity", dino),
    ]
    for ax, (label, values) in zip(axes, metrics):
        ax.plot(actual, values, marker="o")
        ax.set_xlabel("Actual stored bytes mean")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs Actual Bytes")
        ax.grid(True, alpha=0.3)
    quality_plot = analysis_dir / "actual_bytes_vs_quality_metrics.png"
    fig.tight_layout()
    fig.savefig(quality_plot, dpi=160)
    plt.close(fig)

    baseline = summary.get("baseline_reference", {})
    baseline_bytes = baseline.get("text_only_real", {}).get("mean_stored_bytes", {})
    baseline_comparison = []
    for row in rows:
        budget_key = str(row["nominal_budget_bytes"])
        baseline_actual = baseline_bytes.get(budget_key)
        baseline_comparison.append(
            {
                "nominal_budget_bytes": int(row["nominal_budget_bytes"]),
                "current_actual_stored_bytes_mean": float(row["actual_stored_bytes_mean"]),
                "baseline_actual_stored_bytes_mean": float(baseline_actual) if baseline_actual is not None else None,
                "delta_vs_baseline_bytes": (
                    float(row["actual_stored_bytes_mean"]) - float(baseline_actual)
                    if baseline_actual is not None
                    else None
                ),
            }
        )

    analysis = {
        "summary_path": str(summary_path),
        "analysis_generated_at": utcnow_iso(),
        "curve_classification": classify_curve(rows),
        "full_subset_used": bool(summary.get("full_subset_used", False)),
        "nominal_budget_bytes": nominal,
        "actual_stored_bytes_mean": actual,
        "actual_byte_utilization_ratio": utilization,
        "semantic_similarity_mean": semantic,
        "scene_qa_accuracy_mean": qa,
        "dino_similarity_mean": dino,
        "baseline_comparison": baseline_comparison,
        "plot_paths": {
            "nominal_budget_vs_actual_bytes": str(budget_plot),
            "nominal_budget_vs_utilization_ratio": str(utilization_plot),
            "actual_bytes_vs_quality_metrics": str(quality_plot),
        },
    }
    analysis_path = analysis_dir / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    return analysis


def cmd_build(args: argparse.Namespace) -> int:
    username = resolve_username(args.username)
    slug = args.slug or DEFAULT_SLUG
    title = args.title or DEFAULT_TITLE
    build_dir = build_kernel_tree(
        build_dir=args.build_dir,
        username=username,
        slug=slug,
        title=title,
        private=not args.public,
        internet=not args.no_internet,
    )
    print(f"Built Kaggle kernel package at {build_dir}")
    print(f"Kernel ref: {kernel_ref(username, slug)}")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    require_kaggle_cli()
    username = resolve_username(args.username)
    env = kaggle_env(username)
    slug = args.slug or DEFAULT_SLUG
    title = args.title or DEFAULT_TITLE
    build_dir = build_kernel_tree(
        build_dir=args.build_dir,
        username=username,
        slug=slug,
        title=title,
        private=not args.public,
        internet=not args.no_internet,
    )
    command = [
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(build_dir),
        "--accelerator",
        args.accelerator,
        "-t",
        str(args.timeout_seconds),
    ]
    result = run_command(command, capture_output=True, env=env)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        return result.returncode
    print(f"Submitted {kernel_ref(username, slug)}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    require_kaggle_cli()
    username = resolve_username(args.username)
    slug = args.slug or DEFAULT_SLUG
    env = kaggle_env(username)
    result = run_command(
        ["kaggle", "kernels", "status", kernel_ref(username, slug)],
        capture_output=True,
        env=env,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    return result.returncode


def wait_for_terminal_status(username: str, slug: str, poll_seconds: int, timeout_seconds: int) -> tuple[str, str]:
    deadline = time.time() + timeout_seconds
    while True:
        status, output, _ = query_remote_status(username, slug)
        print(f"[status] {status}")
        terminal = status.lower()
        if any(token in terminal for token in ["complete", "success", "failed", "error", "cancel"]):
            return status, output
        if time.time() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for Kaggle kernel {kernel_ref(username, slug)} after {timeout_seconds} seconds."
            )
        time.sleep(poll_seconds)


def cmd_wait(args: argparse.Namespace) -> int:
    require_kaggle_cli()
    username = resolve_username(args.username)
    slug = args.slug or DEFAULT_SLUG
    status, output = wait_for_terminal_status(username, slug, args.poll_seconds, args.timeout_seconds)
    print(output, end="" if output.endswith("\n") else "\n")
    return 0 if any(token in status.lower() for token in ["complete", "success"]) else 1


def cmd_download(args: argparse.Namespace) -> int:
    require_kaggle_cli()
    username = resolve_username(args.username)
    slug = args.slug or DEFAULT_SLUG
    env = kaggle_env(username)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_command(
        [
            "kaggle",
            "kernels",
            "output",
            kernel_ref(username, slug),
            "-p",
            str(args.output_dir),
            "-o",
        ],
        capture_output=True,
        env=env,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    return result.returncode


def cmd_report(args: argparse.Namespace) -> int:
    result = load_result_json(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    status = str(result.get("status", "")).upper()
    return 0 if status == "PASS" else 1


def cmd_run(args: argparse.Namespace) -> int:
    submit_code = cmd_submit(args)
    if submit_code != 0:
        return submit_code
    wait_code = cmd_wait(args)
    download_code = cmd_download(args)
    if download_code != 0:
        return download_code
    report_code = cmd_report(args)
    if wait_code != 0:
        return wait_code
    return report_code


def new_experiment_state(args: argparse.Namespace, *, existing_slug: str | None = None) -> dict[str, object]:
    experiment_id = args.experiment_id
    submitted_at = utcnow_iso()
    git_commit = args.git_commit or git_commit_short()
    slug = existing_slug or args.slug or make_experiment_slug(experiment_id, git_commit, submitted_at)
    username = resolve_username(args.username)
    build_dir = default_build_dir_for_experiment(experiment_id)
    output_dir = default_output_dir_for_experiment(experiment_id)
    analysis_dir = default_analysis_dir_for_experiment(experiment_id)
    title = args.title or slug
    return {
        "experiment_id": experiment_id,
        "kaggle_slug": slug,
        "kernel_ref": kernel_ref(username, slug),
        "git_commit": git_commit,
        "submitted_at": submitted_at,
        "state": "CREATED",
        "last_checked_at": None,
        "output_directory": str(output_dir),
        "analysis_directory": str(analysis_dir),
        "build_directory": str(build_dir),
        "username": username,
        "title": title,
        "accelerator": args.accelerator,
        "is_private": not args.public,
        "internet_enabled": not args.no_internet,
    }


def cmd_submit_experiment(args: argparse.Namespace) -> int:
    require_kaggle_cli()
    ensure_state_dirs()
    if state_path(args.experiment_id).exists():
        raise SystemExit(
            f"Experiment state already exists for {args.experiment_id}. "
            "Use `resume-experiment` or choose a new experiment ID."
        )

    state = new_experiment_state(args)
    write_state(state)

    build_dir = build_kernel_tree(
        build_dir=Path(str(state["build_directory"])),
        username=str(state["username"]),
        slug=str(state["kaggle_slug"]),
        title=str(state["title"]),
        private=bool(state["is_private"]),
        internet=bool(state["internet_enabled"]),
    )
    env = kaggle_env(str(state["username"]))
    command = [
        "kaggle",
        "kernels",
        "push",
        "-p",
        str(build_dir),
        "--accelerator",
        str(state["accelerator"]),
        "-t",
        str(args.timeout_seconds),
    ]
    result = run_command(command, capture_output=True, env=env)
    state["submit_stdout"] = result.stdout
    state["submit_stderr"] = result.stderr
    state["state"] = "SUBMITTED" if result.returncode == 0 else "FAILED"
    state["submit_returncode"] = result.returncode
    if result.returncode == 0:
        state = sync_remote_state(state)
    write_state(state)

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    print(json.dumps(state, indent=2, sort_keys=True))
    return result.returncode


def cmd_register_existing(args: argparse.Namespace) -> int:
    ensure_state_dirs()
    if state_path(args.experiment_id).exists():
        raise SystemExit(
            f"Experiment state already exists for {args.experiment_id}. "
            "Use `resume-experiment` instead."
        )
    state = new_experiment_state(args, existing_slug=args.slug)
    state["submission_mode"] = "adopt_existing_remote_job"
    state["state"] = "SUBMITTED"
    state = sync_remote_state(state)
    path = write_state(state)
    print(f"Registered existing job state at {path}")
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


def cmd_status_experiment(args: argparse.Namespace) -> int:
    state = sync_remote_state(load_state(args.experiment_id))
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0 if str(state["state"]) != "FAILED" else 1


def cmd_download_experiment(args: argparse.Namespace, state: dict[str, object] | None = None) -> int:
    require_kaggle_cli()
    state = state or load_state(args.experiment_id)
    output_dir = Path(str(state["output_directory"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    env = kaggle_env(str(state["username"]))
    result = run_command(
        [
            "kaggle",
            "kernels",
            "output",
            str(state["kernel_ref"]),
            "-p",
            str(output_dir),
            "-o",
        ],
        capture_output=True,
        env=env,
    )
    state["download_returncode"] = result.returncode
    state["downloaded_at"] = utcnow_iso()
    if result.returncode == 0:
        state["state"] = "RESULTS_DOWNLOADED"
    write_state(state)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    return result.returncode


def cmd_analyze_experiment(args: argparse.Namespace, state: dict[str, object] | None = None) -> int:
    state = state or load_state(args.experiment_id)
    output_dir = Path(str(state["output_directory"]))
    analysis_dir = Path(str(state["analysis_directory"]))
    analysis = analyze_sweep_outputs(output_dir, analysis_dir)
    state["analysis_generated_at"] = analysis["analysis_generated_at"]
    state["analysis_json"] = str(analysis_dir / "analysis.json")
    state["state"] = "ANALYZED"
    write_state(state)
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


def cmd_resume_experiment(args: argparse.Namespace) -> int:
    state = sync_remote_state(load_state(args.experiment_id))
    print(f"[remote] {state['kernel_ref']} -> {state['remote_status_text']}")
    if state["state"] == "RUNNING_REMOTE":
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0
    if state["state"] == "FAILED":
        print(json.dumps(state, indent=2, sort_keys=True))
        return 1
    if state["state"] == "RESULTS_DOWNLOADED" and not args.no_analyze:
        return cmd_analyze_experiment(args, state=state)
    if state["state"] == "COMPLETED":
        download_code = cmd_download_experiment(args, state=state)
        if download_code != 0:
            return download_code
        if args.no_analyze:
            refreshed = load_state(args.experiment_id)
            print(json.dumps(refreshed, indent=2, sort_keys=True))
            return 0
        return cmd_analyze_experiment(args, state=load_state(args.experiment_id))
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


def cmd_list_experiments(_: argparse.Namespace) -> int:
    ensure_state_dirs()
    rebuild_state_index()
    print(STATE_INDEX_PATH.read_text(), end="")
    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--username", default=None, help="Kaggle username. Defaults to KAGGLE_USERNAME or PrayYoung.")
    parser.add_argument("--slug", default=None, help="Kaggle kernel slug.")
    parser.add_argument("--title", default=None, help="Kaggle kernel title.")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR, help="Local staging directory for the kernel package.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Local directory for downloaded kernel outputs.")
    parser.add_argument("--accelerator", default=DEFAULT_ACCELERATOR, help="Kaggle accelerator ID.")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="Kaggle run timeout, and wait timeout when applicable.")
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS, help="Polling interval for wait.")
    parser.add_argument("--public", action="store_true", help="Create or update a public kernel instead of private.")
    parser.add_argument("--no-internet", action="store_true", help="Disable internet for the Kaggle kernel.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and run the Kaggle remote GPU smoke kernel.")
    add_common_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build", help="Stage the Kaggle kernel locally.")
    subparsers.add_parser("submit", help="Build and submit the Kaggle kernel.")
    subparsers.add_parser("status", help="Query the latest Kaggle run status.")
    subparsers.add_parser("wait", help="Poll until the latest Kaggle run reaches a terminal state.")
    subparsers.add_parser("download", help="Download the latest kernel outputs.")
    subparsers.add_parser("report", help="Read output/result.json and return PASS or FAIL.")
    subparsers.add_parser("run", help="Submit, wait, download, and report PASS or FAIL.")

    submit_experiment = subparsers.add_parser(
        "submit-experiment",
        help="Submit a uniquely identified Kaggle experiment and persist local state without waiting.",
    )
    submit_experiment.add_argument("--experiment-id", required=True, help="Stable local experiment identifier.")
    submit_experiment.add_argument("--git-commit", default=None, help="Override git commit stored with this experiment.")

    register_existing = subparsers.add_parser(
        "register-existing",
        help="Adopt an already-submitted Kaggle job into local state tracking without resubmitting.",
    )
    register_existing.add_argument("--experiment-id", required=True, help="Stable local experiment identifier.")
    register_existing.add_argument("--git-commit", default=None, help="Override git commit stored with this experiment.")

    status_experiment = subparsers.add_parser(
        "status-experiment",
        help="Refresh and print the saved state for an existing experiment.",
    )
    status_experiment.add_argument("--experiment-id", required=True, help="Stable local experiment identifier.")

    resume_experiment = subparsers.add_parser(
        "resume-experiment",
        help="Resume an existing experiment by checking remote state, downloading results, and analyzing if complete.",
    )
    resume_experiment.add_argument("--experiment-id", required=True, help="Stable local experiment identifier.")
    resume_experiment.add_argument("--no-analyze", action="store_true", help="Download completed outputs but skip local analysis.")

    analyze_experiment = subparsers.add_parser(
        "analyze-experiment",
        help="Analyze already-downloaded sweep outputs for a saved experiment.",
    )
    analyze_experiment.add_argument("--experiment-id", required=True, help="Stable local experiment identifier.")

    list_experiments = subparsers.add_parser(
        "list-experiments",
        help="Print the machine-readable registry of locally tracked Kaggle experiments.",
    )
    list_experiments.set_defaults(command="list-experiments")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    command_map = {
        "build": cmd_build,
        "submit": cmd_submit,
        "status": cmd_status,
        "wait": cmd_wait,
        "download": cmd_download,
        "report": cmd_report,
        "run": cmd_run,
        "submit-experiment": cmd_submit_experiment,
        "register-existing": cmd_register_existing,
        "status-experiment": cmd_status_experiment,
        "resume-experiment": cmd_resume_experiment,
        "analyze-experiment": cmd_analyze_experiment,
        "list-experiments": cmd_list_experiments,
    }
    return command_map[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
