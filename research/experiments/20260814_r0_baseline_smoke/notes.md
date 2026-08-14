# 20260814 R0 Baseline Smoke

Date: Friday, August 14, 2026

## Goal

Classify the frozen baseline commit `356dc691381feaf7c269801cb484caaa24fe486c` under the recovered Kaggle protocol without mixing in repaired-branch behavior.

## Frozen controls

- source commit under test: `356dc691381feaf7c269801cb484caaa24fe486c`
- task: `lift`
- dataset: RoboMimic `ph` / `low_dim`
- smoke training overrides:
  - `task=lift`
  - `max_steps=20`
  - `val_every=20`
  - `log_every=10`
  - `batch_size=16`
  - `num_workers=0`
  - `run_dir=/kaggle/working/runs`
  - `exp_name=wm_lift_smoke`

## Retry chain

1. `20260814_r0_baseline_smoke`
   - failure class: engineering
   - cause: the historical source bundle did not include `requirements-kaggle.txt`, so the recovered Kaggle wrapper could not install its pinned runtime.
2. `20260814_r0_baseline_smoke_retry1`
   - failure class: baseline code
   - cause: the frozen baseline reached world-model training and failed in `src/blind_diffusion/data/robomimic_dataset.py`
   - exact first truthful failure:

```text
TypeError: 'NoneType' object is not subscriptable
  File "/tmp/blind-diffusion/src/blind_diffusion/data/robomimic_dataset.py", line 31, in __init__
    demo_names = sorted(list(self._h5["data"].keys()))
```

## Interpretation

- `R0` is now classified.
- The frozen baseline is not executable under this protocol because the original dataset constructor dereferences `self._h5` before opening the HDF5 file.
- This means any post-repair executable result must be reported as a repaired-branch reference result, not as a baseline result.

## Next step

Run the repaired branch under the same frozen low-dimensional protocol to establish the first executable post-repair reference point.
