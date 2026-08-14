# 20260814 R1 Repaired Reference Smoke

Date: Friday, August 14, 2026

## Goal

Establish the first executable repaired-branch reference point after `R0` showed that the frozen baseline is not runnable.

## Protocol

- source branch: `codex/blind-diffusion-research-recovery`
- task: `lift`
- dataset: RoboMimic `ph` / `low_dim`
- world-model smoke:
  - `task=lift`
  - `max_steps=20`
  - `val_every=20`
  - `log_every=10`
  - `batch_size=16`
  - `num_workers=0`
  - `run_dir=/kaggle/working/runs`
  - `exp_name=wm_lift_smoke`
- low-dimensional MPC eval smoke:
  - `eval.mode=mpc`
  - `task=lift`
  - `episodes=2`
  - `max_steps=200`
  - `run_dir=/kaggle/working/eval_runs`
  - `exp_name=eval_lift_smoke`

## Expectations

- dataset download succeeds
- regression tests pass
- repaired branch completes world-model smoke training
- repaired branch completes low-dimensional MPC evaluation and writes structured metrics

## Interpretation guardrail

This run is only an executable repaired-branch reference point. It is not a clean-vs-blind robustness benchmark and not a final quality claim.

## Retry chain on Friday, August 14, 2026

1. `20260814_r1_repaired_reference_smoke`
   - failure class: engineering
   - world-model smoke training completed
   - low-dimensional evaluation failed in `src/blind_diffusion/env/robomimic_env.py`
   - exact first truthful failure:

```text
TypeError: create_env_from_metadata() got an unexpected keyword argument 'use_image_obj'
```

## Planned retry

- fix the `robomimic` compatibility keyword from `use_image_obj` to `use_image_obs`
- preserve the same train and eval protocol
- submit with a new run identity

2. `20260814_r1_repaired_reference_smoke_retry1`
   - failure class: engineering
   - dataset download passed
   - regression tests passed
   - world-model smoke training passed
   - low-dimensional evaluation failed during environment import because the Kaggle dependency stack resolved to `robomimic 0.3.0`, which imports `mujoco_py`

## Planned retry 2

- pin the Kaggle and project dependency stack to `robomimic >= 0.4.0, < 0.5` and `robosuite >= 1.5.1, < 1.6`
- keep the same repaired-branch train and eval protocol
- submit with a new run identity

3. `20260814_r1_repaired_reference_smoke_retry2`
   - failure class: engineering
   - dependency install failed before runtime
   - Kaggle's PyPI view exposed only `robomimic` versions `0.1.0`, `0.2.0`, and `0.3.0`
   - `robomimic >= 0.4.0` was therefore not a valid pip-only repair path

## Planned retry 3

- keep `requirements-kaggle.txt` for core wheels only
- install `robomimic` from Git tag `v0.4`
- install `robosuite` from Git tag `v1.5.1`
- preserve the same repaired-branch train and eval protocol
- submit with a new run identity
