# Blind-Diffusion Research State

Last updated: 2026-08-14 UTC

## Current repository state

- Local repository is the source of truth.
- HEAD commit: `356dc691381feaf7c269801cb484caaa24fe486c`
- Branch at freeze time: `master`
- Active research branch: `codex/blind-diffusion-research-recovery`
- Working tree at freeze time: clean

## Current codebase map

- RSSM: `src/blind_diffusion/models/rssm.py`
- Low-dimensional MPC eval: `src/blind_diffusion/eval/evaluate_lowdim.py`
- Diffusion eval: `src/blind_diffusion/eval/evaluate_diffusion.py`
- Image eval and existing sensor block path: `src/blind_diffusion/eval/evaluate_image.py`
- Diffusion sampler: `src/blind_diffusion/diffusion/diffusion.py`
- Low-dimensional dataset: `src/blind_diffusion/data/robomimic_dataset.py`
- Image dataset: `src/blind_diffusion/data/robomimic_dataset_image.py`
- Metrics: `src/blind_diffusion/utils/metrics.py`
- Config entrypoints: `src/blind_diffusion/configs/*.yaml`

## Frozen baseline metadata

- Baseline commit: `356dc691381feaf7c269801cb484caaa24fe486c`
- Baseline configs:
  - `src/blind_diffusion/configs/eval_mpc.yaml`
  - `src/blind_diffusion/configs/eval_diffusion.yaml`
  - `src/blind_diffusion/configs/eval_image.yaml`
- Baseline execution artifacts: none discovered yet in the local repository
- Baseline checkpoints discovered locally: none discovered yet in the local repository
- Local runtime discovered: `.venv/bin/python` exists, local `uv` binary does not
- Kaggle remote workflow recovered from: `/Users/peiyan1/Desktop/ai_agent_memory/scripts/kaggle_remote_smoke.py`
- Local executability on Friday, August 14, 2026:
  - `ROBO_DATA` unset
  - no `runs/` directory
  - configured checkpoints absent
  - `.venv` lacks `hydra-core`
  - `.venv` lacks `robomimic`
  - local CUDA unavailable

## Hypotheses

1. `H1`: low-dimensional MPC evaluation resets RSSM memory incorrectly every environment step.
2. `H2`: both RoboMimic dataset classes are broken during initialization and unsafe for multi-worker loading.
3. `H3`: RSSM lacks a reusable single-step prior transition primitive even though image eval has an inline prior-rollout path.
4. `H4`: diffusion evaluation still uses only full-step DDPM sampling; DDIM and latency instrumentation are absent.

## Confirmed bugs

1. Dataset initialization bug is present in both dataset classes.
   - Both constructors set `self._h5 = None` and then immediately access `self._h5["data"]`.
   - This is a hard failure before any DataLoader worker behavior is tested.
2. Low-dimensional planner memory bug is very likely present and already code-confirmed at the control-loop level.
   - Episode start initializes a posterior state once.
   - Subsequent steps call `wm.rssm.observe(...)` without carrying forward the previous `h` and `z`.
   - That means the posterior at step `t+1` is conditioned only on the latest observation and action tensor passed into a fresh zero-state rollout, not on the causal latent trajectory.

3. RSSM did not expose a reusable single-step prior rollout primitive.
   - Image evaluation had an inline equivalent.
   - Core RSSM did not.

## Confirmed repairs

1. Added `RSSM.imagine_step(...)` and verified it matches the first step of `RSSM.imagine(...)` under fixed RNG.
2. Reworked both RoboMimic datasets to inspect metadata in short-lived file contexts and lazily open per-process HDF5 handles.
3. Repaired low-dimensional evaluation to carry recurrent RSSM state causally via `observe_step(...)` and the previous action.
4. Added focused local regression tests:
   - `tests/test_rssm.py`
   - `tests/test_datasets.py`

## Unconfirmed claims

1. DDIM can improve latency materially without unacceptable quality loss.
2. RSSM single-step prior rollout improves blackout handling in low-dimensional control.
3. Antigravity-reported success-rate and latency gains are real.
4. Existing image-eval sensor-block behavior is scientifically sound and reproducible.

## Experiment queue

1. `R0`: baseline executability and artifact recovery
2. `R1`: recurrent-state persistence diagnostic and correction-only ablation
3. `R2`: reusable RSSM imagine-step prior-only rollout
4. `R6`: dataset construction and worker stress test at `num_workers = 0, 1, 2, 4`
5. `R3`: sensor blackout benchmark after correctness fixes
6. `R4`: controlled noise robustness benchmark
7. `R5`: DDPM vs DDIM latency and quality Pareto benchmark

## Active experiment

- `archaeology-and-local-diagnostics`
- Goal: freeze baseline, verify correctness claims locally, and recover the existing Kaggle workflow

## Completed experiments

- `20260814_local_diagnostics`
  - Baseline classified as not executable locally as configured.
  - Two core Antigravity bug claims reproduced locally.
  - Local regression tests now pass for RSSM stepping and dataset lifecycle.

## Best result so far

- No validated performance result yet.
- Best evidence so far is negative engineering evidence: the current repository state still contains at least one hard dataset-construction bug and one likely planner-memory bug.

## Blockers

1. No baseline checkpoints or prior run artifacts have been found in the local repository yet.
2. Dataset paths and local RoboMimic assets are not yet confirmed.
3. Local environment still lacks `hydra-core` and `robomimic`.
4. Local machine has no CUDA device, so latency and control-quality claims still require remote execution.

## Next action

1. Decide whether to vendor the recovered Kaggle workflow into this repository or invoke it directly from the external recovered path.
2. Enable structured experiment output under `research/experiments/` for actual baseline and ablation runs.
3. Reproduce `R0` remotely with frozen baseline metadata before broader changes like metrics expansion or DDIM.
