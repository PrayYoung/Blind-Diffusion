# 20260814 Local Diagnostics

Date: Friday, August 14, 2026

## Purpose

Freeze the repository state, classify whether the current baseline is runnable locally, and verify or refute the highest-priority correctness hypotheses before any broader experimentation.

## Frozen baseline

- Commit: `356dc691381feaf7c269801cb484caaa24fe486c`
- Configs:
  - `src/blind_diffusion/configs/eval_mpc.yaml`
  - `src/blind_diffusion/configs/eval_diffusion.yaml`
  - `src/blind_diffusion/configs/eval_image.yaml`

## Baseline executability result

- Not executable locally as configured.
- Reasons:
  - `ROBO_DATA` was unset.
  - No `runs/` directory or configured checkpoints were present in the repository.
  - The repo venv lacked `hydra-core` and `robomimic`.
  - No local CUDA device was available.

## Confirmed issues

1. Both RoboMimic dataset constructors were broken before any worker logic:
   - low-dimensional dataset
   - image dataset
   - failure: `TypeError: 'NoneType' object is not subscriptable`
2. Low-dimensional evaluation did not carry RSSM latent state causally between environment steps.
3. RSSM lacked a reusable prior-only single-step transition primitive.

## Verification after fixes

- `PYTHONPATH=src ./.venv/bin/python -m unittest tests.test_rssm`
  - passed
- `PYTHONPATH=src ./.venv/bin/python -m unittest tests.test_datasets`
  - passed

## Interpretation

- Two Antigravity bug claims were reproduced locally and fixed:
  - dataset lifecycle bug
  - planner-memory bug
- Local environment and asset gaps still prevent a faithful RoboMimic baseline run on this machine alone.
- Next high-value step is remote baseline reproduction using the recovered Kaggle workflow, while keeping the benchmark protocol frozen.
