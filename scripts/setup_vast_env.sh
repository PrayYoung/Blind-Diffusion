#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="blind_diffusion_py310"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required in the Vast image to provision the Python 3.10 runtime" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10
fi

conda activate "$ENV_NAME"

apt-get update
apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-input --no-cache-dir -r requirements-vast.txt
python -m pip install --no-input --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0

# Keep this exactly aligned with the project requirement. A floating <4.0
# install can select a newer MuJoCo, then fail editable installation when
# setuptools validates the project's strict mujoco==3.5.0 requirement.
python -m pip install --no-input --no-cache-dir "mujoco==3.5.0" "robosuite==1.4.1"
python -m pip install --no-input --no-cache-dir "robomimic==0.3.0" --no-deps

# Install Blind-Diffusion last so remote code uses the transferred snapshot
# without letting pip re-resolve the robomimic / robosuite stack.
python -m pip install --no-input --no-cache-dir --no-deps -e .
