#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install --no-input --no-cache-dir -r requirements-vast.txt

# Match the reference wrapper-based stack before adding the project itself.
python3 -m pip install --no-input --no-cache-dir "mujoco>=3.1.0" "robosuite==1.4.1"
python3 -m pip install --no-input --no-cache-dir "robomimic==0.3.0" --no-deps

# Install Blind-Diffusion last so local source changes are visible remotely
# without letting pip re-resolve the robomimic / robosuite stack.
python3 -m pip install --no-input --no-cache-dir --no-deps -e .
