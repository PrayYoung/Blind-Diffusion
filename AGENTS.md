# Blind-Diffusion Session Guide

This repository is being recovered as a research project, not a patch task.

## Research branch and baseline

- Preserve the frozen baseline commit before correctness-affecting edits:
  - `356dc691381feaf7c269801cb484caaa24fe486c`
- Active recovery branch:
  - `codex/blind-diffusion-research-recovery`
- Do not relabel repaired results as baseline results.

## Canonical state files

- Machine-readable state: `RESEARCH_STATE.json`
- Human-readable state: `RESEARCH_STATE.md`
- Structured experiment artifacts should live under:
  - `research/experiments/<experiment_id>/`

## Current high-priority findings

1. `src/blind_diffusion/eval/evaluate_lowdim.py` appears to break causal RSSM state propagation across environment steps.
2. `src/blind_diffusion/data/robomimic_dataset.py` and `src/blind_diffusion/data/robomimic_dataset_image.py` both access `self._h5` before opening the HDF5 file.
3. `src/blind_diffusion/diffusion/diffusion.py` has DDPM sampling only; no DDIM path exists yet.
4. `src/blind_diffusion/eval/evaluate_image.py` contains inline sensor-block logic and an ad hoc prior-rollout path, but there is no reusable modality-aware corruption module yet.

## Compute backend

- Reuse the existing Kaggle workflow already present on this machine rather than inventing a new one.
- Recovered reference implementation:
  - `/Users/peiyan1/Desktop/ai_agent_memory/scripts/kaggle_remote_smoke.py`
- Relevant traits from that workflow:
  - repo-local Kaggle config support
  - persisted job state
  - persisted output and analysis directories
  - kernel slug generation tied to experiment id and git commit

## Evidence standard

- Separate:
  - correctness repairs
  - research changes
  - infrastructure changes
- Run the smallest discriminating experiment first.
- Record negative results plainly.
- Do not change the benchmark protocol after seeing results without versioning the protocol and rerunning both baseline and candidate.

## Immediate next tasks

1. Confirm local dataset and checkpoint availability.
2. Run baseline executability checks before code repairs.
3. Add focused regression tests for:
   - recurrent-state persistence
   - RSSM prior-only step semantics
   - HDF5 dataset construction and worker safety
   - structured metrics consistency
