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
