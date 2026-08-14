# 20260814 Remote Smoke Preparation

Date: Friday, August 14, 2026

## Goal

Prepare the recovered Kaggle GPU workflow for Blind-Diffusion and submit a first remote smoke that:

1. installs Blind-Diffusion runtime dependencies,
2. downloads the public RoboMimic `lift` low-dimensional dataset,
3. runs the local regression tests,
4. runs a tiny world-model training smoke.

## Local preparation status

- `scripts/kaggle_remote_smoke.py`
  - vendored from the user's existing workflow and retargeted to Blind-Diffusion
  - packages the current repository dynamically at build time
- `kaggle/remote_smoke/remote_smoke.py`
  - rewritten as a Blind-Diffusion-specific smoke payload
- `requirements-kaggle.txt`
  - rewritten for Blind-Diffusion dependencies

## Local validation

- Python bytecode compilation passed.
- Local Kaggle kernel package build passed.
- Built kernel ref:
  - `PrayYoung/blind-diffusion-remote-smoke`

## Blocker

- Submission did not proceed because explicit user approval is required before exporting the repository snapshot to Kaggle.
