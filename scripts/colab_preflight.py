"""Read-only inventory before borrowing an existing Colab runtime."""
import importlib.metadata
import json
import shutil
import subprocess
import sys

import torch

packages = {}
for package in ("torch", "numpy", "hydra-core", "h5py", "robomimic", "robosuite", "mujoco"):
    try:
        packages[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        packages[package] = None
inventory = {
    "python": sys.version.split()[0], "packages": packages,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "free_gpu_bytes": torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else None,
    "free_disk_bytes": shutil.disk_usage("/content").free,
}
processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                           capture_output=True, text=True, check=False)
inventory["gpu_processes"] = processes.stdout.strip().splitlines()
print(json.dumps(inventory, indent=2))
