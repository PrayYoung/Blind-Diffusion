#!/usr/bin/env bash
set -euo pipefail

# RunPod's selected PyTorch image already provides Python 3.10 but not Conda.
# Keep this runtime reconstruction equivalent to the Vast setup without adding
# a second environment manager.
python - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"Python 3.10 is required, found {sys.version}")
PY

# Do not use RunPod's CUDA-version placement filter: it is an exact allowed
# version list, not a "12.4 or newer" constraint.  Check the real host driver
# here instead, before spending time on dependency installation.  CUDA 12.4's
# standard Linux driver baseline is 550.54.14.
python - <<'PY'
import subprocess

required = (550, 54, 14)
raw = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
    text=True,
).splitlines()[0].strip()
actual = tuple(int(part) for part in raw.split(".")[:3])
if actual < required:
    raise SystemExit(
        f"NVIDIA driver {raw} is too old for the CUDA 12.4 PyTorch wheel; "
        f"need at least {'.'.join(map(str, required))}"
    )
print(f"CUDA 12.4 driver check passed: {raw}")
PY

apt-get update
apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-input --no-cache-dir -r requirements-vast.txt
python -m pip install --no-input --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0

# robosuite 1.4.1's controller implementation reads MjData.qM, which is
# available in the MuJoCo 2.3 API but removed in MuJoCo 3.x. Python 3.10 has
# a prebuilt 2.3.7 wheel, unlike the retired Kaggle Python 3.12 route.
python -m pip install --no-input --no-cache-dir "mujoco==2.3.7" "robosuite==1.4.1"
python -m pip install --no-input --no-cache-dir "robomimic==0.3.0" --no-deps
python -m pip install --no-input --no-cache-dir --no-deps -e .
