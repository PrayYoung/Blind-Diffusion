from __future__ import annotations

import base64
import gzip
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback


WORKING_ROOT = Path("/kaggle/working")
OUTPUT_DIR = WORKING_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPO_DIR = Path(tempfile.gettempdir()) / "blind-diffusion"

TORCH_VERSION = "2.6.0"
TORCHVISION_VERSION = "0.21.0"
TORCHAUDIO_VERSION = "2.6.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu118"
BUNDLED_REPO_GZ_B64 = "__BUNDLED_REPO_GZ_B64__"
DATASET_ROOT = WORKING_ROOT / "robomimic_datasets"


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _pip_install(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return _run([sys.executable, "-m", "pip", "install", "--no-cache-dir", *args], cwd=cwd)


def _materialize_repo_bundle() -> dict[str, object]:
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    files = json.loads(gzip.decompress(base64.b64decode(BUNDLED_REPO_GZ_B64)).decode("utf-8"))
    for relative, content in files.items():
        path = REPO_DIR / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return {
        "repo_dir": str(REPO_DIR),
        "repo_source": "bundled_snapshot",
        "file_count": len(files),
    }


def _install_runtime() -> dict[str, object]:
    _run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
    torch_install = _pip_install(
        [
            "--index-url",
            TORCH_INDEX_URL,
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
            f"torchaudio=={TORCHAUDIO_VERSION}",
        ]
    )
    if torch_install.returncode != 0:
        raise RuntimeError(
            "Pinned torch install failed.\n"
            f"stdout:\n{torch_install.stdout}\n"
            f"stderr:\n{torch_install.stderr}"
        )

    req_install = _pip_install(["-r", "requirements-kaggle.txt"], cwd=REPO_DIR)
    if req_install.returncode != 0:
        raise RuntimeError(
            "requirements-kaggle install failed.\n"
            f"stdout:\n{req_install.stdout}\n"
            f"stderr:\n{req_install.stderr}"
        )

    editable_install = _pip_install(['-e', '.[robomimic]'], cwd=REPO_DIR)
    if editable_install.returncode != 0:
        raise RuntimeError(
            "Editable project install failed.\n"
            f"stdout:\n{editable_install.stdout}\n"
            f"stderr:\n{editable_install.stderr}"
        )

    return {
        "torch_install_returncode": torch_install.returncode,
        "requirements_install_returncode": req_install.returncode,
        "editable_install_returncode": editable_install.returncode,
    }


def _probe_runtime() -> dict[str, object]:
    import h5py
    import hydra
    import numpy
    import omegaconf
    import robomimic
    import torch

    info: dict[str, object] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "hydra": hydra.__version__,
        "omegaconf": omegaconf.__version__,
        "h5py": h5py.__version__,
        "numpy": numpy.__version__,
        "robomimic": getattr(robomimic, "__version__", "unknown"),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "gpu_memory_bytes": props.total_memory,
                "compute_capability": f"sm_{props.major}{props.minor}",
            }
        )
    return info


def _download_dataset() -> dict[str, object]:
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    result = _run(
        [
            sys.executable,
            "-m",
            "robomimic.scripts.download_datasets",
            "--tasks",
            "lift",
            "--dataset_types",
            "ph",
            "--hdf5_types",
            "low_dim",
            "--download_dir",
            str(DATASET_ROOT),
        ],
        cwd=REPO_DIR,
    )
    downloaded_path = DATASET_ROOT / "lift" / "ph" / "low_dim_v141.hdf5"
    normalized_path = DATASET_ROOT / "lift" / "low_dim_v141.hdf5"
    if downloaded_path.exists():
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        if not normalized_path.exists():
            shutil.copy2(downloaded_path, normalized_path)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "dataset_path": str(normalized_path),
        "downloaded_path": str(downloaded_path),
        "dataset_exists": normalized_path.exists(),
    }


def _run_unittests() -> dict[str, object]:
    tests_dir = REPO_DIR / "tests"
    if not tests_dir.exists():
        return {
            "skipped": True,
            "reason": "tests directory absent in bundled source snapshot",
        }
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = _run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
        cwd=REPO_DIR,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Unit tests failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return {
        "skipped": False,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _run_world_model_smoke() -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["ROBO_DATA"] = str(DATASET_ROOT)
    result = _run(
        [
            sys.executable,
            "scripts/train_wm.py",
            "task=lift",
            "max_steps=20",
            "val_every=20",
            "log_every=10",
            "batch_size=16",
            "num_workers=0",
            "run_dir=/kaggle/working/runs",
            "exp_name=wm_lift_smoke",
        ],
        cwd=REPO_DIR,
        env=env,
    )
    metrics_path = WORKING_ROOT / "runs" / "wm_lift_smoke" / "metrics.json"
    checkpoint_path = WORKING_ROOT / "runs" / "wm_lift_smoke" / "checkpoints" / "best.pt"
    if result.returncode != 0:
        raise RuntimeError(
            "World model smoke failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "metrics_path": str(metrics_path),
        "metrics_exists": metrics_path.exists(),
        "checkpoint_exists": checkpoint_path.exists(),
        "checkpoint_path": str(checkpoint_path),
        "metrics": json.loads(metrics_path.read_text()) if metrics_path.exists() else None,
    }


def _run_lowdim_eval(checkpoint_path: str) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["ROBO_DATA"] = str(DATASET_ROOT)
    result = _run(
        [
            sys.executable,
            "scripts/eval.py",
            "eval.mode=mpc",
            "task=lift",
            f"checkpoint={checkpoint_path}",
            "episodes=2",
            "max_steps=200",
            "run_dir=/kaggle/working/eval_runs",
            "exp_name=eval_lift_smoke",
        ],
        cwd=REPO_DIR,
        env=env,
    )
    metrics_path = WORKING_ROOT / "eval_runs" / "eval_lift_smoke" / "metrics.json"
    if result.returncode != 0:
        raise RuntimeError(
            "Low-dimensional evaluation smoke failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "metrics_path": str(metrics_path),
        "metrics_exists": metrics_path.exists(),
        "metrics": json.loads(metrics_path.read_text()) if metrics_path.exists() else None,
    }


def main() -> int:
    result: dict[str, object] = {
        "status": "FAIL",
        "cwd": os.getcwd(),
    }
    try:
        repo_info = _materialize_repo_bundle()
        result["repo"] = repo_info
        install_info = _install_runtime()
        result["install"] = install_info
        runtime_info = _probe_runtime()
        result["runtime"] = runtime_info
        dataset_info = _download_dataset()
        result["dataset"] = dataset_info
        if not dataset_info["dataset_exists"]:
            raise RuntimeError(
                "RoboMimic dataset download did not produce lift/low_dim_v141.hdf5.\n"
                f"stdout:\n{dataset_info['stdout']}\n"
                f"stderr:\n{dataset_info['stderr']}"
            )
        unittest_info = _run_unittests()
        result["unittests"] = unittest_info
        smoke_info = _run_world_model_smoke()
        result["world_model_smoke"] = smoke_info
        eval_info = _run_lowdim_eval(smoke_info["checkpoint_path"])
        result["lowdim_eval_smoke"] = eval_info
        result["status"] = "PASS"
    except Exception as exc:
        result.update(
            {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
