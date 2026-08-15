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

4. `20260814_r1_repaired_reference_smoke_retry3`
   - failure class: engineering
   - modern source stack installed successfully
   - runtime reported:
     - `robomimic 0.4.0`
     - `robosuite 1.5.1`
     - CUDA available on Tesla T4
   - first truthful failure:
     - source-installed downloader produced `low_dim_v15.hdf5`
     - frozen protocol expects `lift/low_dim_v141.hdf5`
   - interpretation:
     - this is a protocol-preservation issue, not a research result
     - silently renaming `v15` to `v141` would conflate different dataset families

## Planned retry 4

- install `robomimic` from source tag `v0.3.0`
- install `robosuite` from source tag `v1.4.1`
- preserve the same repaired-branch train and eval protocol
- submit with a new run identity

5. `20260814_r1_repaired_reference_smoke_retry4`
   - failure class: engineering
   - the source tags now match the frozen `v141` dataset family
   - dataset download passed
   - regression tests passed
   - CUDA runtime was available
   - world-model smoke training passed
   - first truthful remaining blocker:
     - `robomimic.env_robosuite` still imports `mujoco_py`
     - this fails before the actual rollout starts

## Planned retry 5

- keep the same `robomimic 0.3.0` / `robosuite 1.4.1` source-tag stack
- bypass `robomimic.env_robosuite` entirely
- read `env_args` from the HDF5 directly and instantiate `robosuite.make(...)`
- preserve the same repaired-branch train and eval protocol
- submit with a new run identity

6. `20260814_r1_repaired_reference_smoke_retry5`
   - failure class: engineering
   - direct `robosuite.make(...)` path did bypass the `mujoco_py` wrapper dependency
   - first truthful remaining blocker:

```text
TypeError: Lift.__init__() got an unexpected keyword argument 'lite_physics'
```

## Planned retry 6

- keep the same `robomimic 0.3.0` / `robosuite 1.4.1` source-tag stack
- keep the same direct `robosuite.make(...)` adapter
- remove only the incompatible `lite_physics` keyword
- preserve the same repaired-branch train and eval protocol
- submit with a new run identity

7. `20260814_r1_repaired_reference_smoke_retry6`
   - failure class: engineering
   - direct `robosuite.make(...)` now progressed into controller initialization
   - first truthful remaining blocker:

```text
AttributeError: 'MjData' object has no attribute 'qM'
```

   - interpretation:
     - `robosuite 1.4.1` declares `mujoco>=2.3.0`
     - the current project metadata was still allowing much newer MuJoCo 3.x
     - the next and final planned compatibility retry should pin MuJoCo to the `2.3.x` family

## Planned retry 7

- keep the same `robomimic 0.3.0` / `robosuite 1.4.1` source-tag stack
- keep the same direct `robosuite.make(...)` adapter
- pin MuJoCo to `>=2.3.0,<3.0`
- preserve the same repaired-branch train and eval protocol
- submit with a new run identity

8. `20260814_r1_repaired_reference_smoke_retry7`
   - failure class: engineering
   - first truthful remaining blocker moved earlier into dependency installation:

```text
ERROR: Failed to build installable wheels for some pyproject.toml based projects (mujoco)
```

   - interpretation:
     - Kaggle's Python `3.12.13` runtime can install modern native `mujoco` wheels, but not the older `2.3.x` family we tried to force for the direct `robosuite.make(...)` reconstruction
     - this makes the direct-path MuJoCo back-pin a dead end on the target backend
     - the exact `robomimic 0.3.0` / `robosuite 1.4.1` wrapper path is still scientifically preferable anyway

## Redirected next attempt

- restore `create_env_from_metadata(...)` / `EnvRobosuite` semantics in `src/blind_diffusion/env/robomimic_env.py`
- keep `low_dim_v141.hdf5`
- install native DeepMind `mujoco==3.5.0` with `robosuite==1.4.1`
- install `robomimic==0.3.0 --no-deps`
- apply only the smallest local compatibility shim for the unconditional `import mujoco_py` in `robomimic.envs.env_robosuite`
- preserve the same train and eval protocol
- submit with a new run identity

9. `20260814_r1_repaired_reference_smoke_retry8`
   - failure class: engineering
   - the exact wrapper-based path now launches on Kaggle, but the first truthful remaining blocker is pip dependency resolution:

```text
ERROR: Cannot install mujoco<3.0 and >=2.3.0 and mujoco==3.5.0 because these package versions have conflicting dependencies.
```

   - interpretation:
     - the intended compatibility stack is already being installed explicitly
     - the follow-up `pip install -r requirements-kaggle.txt` is unnecessarily re-solving third-party metadata and rejecting the override

## Planned retry 9

- keep the exact wrapper-based `create_env_from_metadata(...)` / `EnvRobosuite` path
- keep native `mujoco==3.5.0`, `robosuite==1.4.1`, and `robomimic==0.3.0 --no-deps`
- add only `--no-deps` to the follow-up requirements and editable repo installs
- preserve the same train and eval protocol
- submit with a new run identity

10. `20260815_r1_repaired_reference_vast_retry1`
   - backend: Vast
   - run id: `vast-20260815T000203Z-8215-99cea1`
   - instance id: `47748496`
   - failure class: engineering
   - meaningful transition:
     - the repaired reference system reached the real workload on Vast under Python `3.10`, native MuJoCo `3.x`, `robomimic 0.3.0`, and `robosuite 1.4.1`
   - first truthful failure:

```text
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

   - repair:
     - install system GL libraries on the remote host
     - explicitly install CUDA PyTorch into the new Python `3.10` environment

11. `20260815_r1_repaired_reference_vast_retry2`
   - backend: Vast
   - run id: `vast-20260815T001025Z-11891-46acc1`
   - failure class: infrastructure
   - first truthful failure:

```text
No acceptable offer under hard ceiling $0.350/hr. Cheapest matching offer was $0.363/hr.
```

   - repair:
     - retry once with a one-off CLI ceiling of `$0.37/hr`
     - keep repository pricing/config files unchanged

12. `20260815_r1_repaired_reference_vast_retry3`
   - backend: Vast
   - run id: `vast-20260815T001057Z-11909-d47a4c`
   - instance id: `47748885`
   - failure class: engineering
   - meaningful transition:
     - the repaired reference system again reached the real workload after the GL / torch repair
   - first truthful failure:

```text
AttributeError: 'NoneType' object has no attribute 'eglQueryString'
```

   - interpretation:
     - headless EGL initialization is still incomplete on the remote host
   - repair:
     - install EGL-related system libraries
     - export `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`

13. `20260815_r1_repaired_reference_vast_retry4`
   - backend: Vast
   - run id: `vast-20260815T001517Z-12567-9caea5`
   - instance id: `47749103`
   - failure class: engineering
   - meaningful transition:
     - provisioning, upload, setup, and workload launch all completed on an `RTX 4090`
     - this confirmed that the repaired reference path is executing the real remote script under the intended Python `3.10` / `robomimic 0.3.0` / `robosuite 1.4.1` / native MuJoCo `3.x` stack
   - first truthful failure:

```text
ModuleNotFoundError: No module named 'matplotlib'
```

   - interpretation:
     - the failure occurs in `robomimic.scripts.download_datasets`, so it is a real workload dependency gap rather than a mock or fallback path
     - this is still an engineering failure and does not justify changing the scientific protocol
   - repair:
     - add `matplotlib` to `requirements-vast.txt`
     - keep the workload, dataset family, task, and model path unchanged

## Planned retry 14

- preserve the same repaired-reference Vast workload
- preserve the same Python `3.10` / `robomimic 0.3.0` / `robosuite 1.4.1` / native MuJoCo `3.x` environment target
- add only the missing `matplotlib` dependency to the repo-specific Vast setup
- submit with a new run identity

14. `20260815_r1_repaired_reference_vast_retry5`
   - backend: Vast
   - run id: `vast-20260815T002237Z-13447-ea2fb7`
   - failure class: engineering
   - meaningful transition:
     - rented host: `332635`
     - location: `Indiana, US`
     - GPU: `RTX 4090`
     - hourly price: `$0.356/hr`
     - instance id: `47749465`
     - runner-managed phases reached: `PROVISIONING` -> `WAITING_FOR_SSH` -> `UPLOADING` -> `SETTING_UP` -> `RUNNING`
     - repository snapshot under test: `c3ede09`
     - change relative to retry 4: add `matplotlib` to `requirements-vast.txt`
      - verified runtime banner:
        - Python `3.10.20`
        - `torch 2.4.0+cu124`
        - CUDA available
        - `mujoco 3.11.0`
        - `robomimic 0.3.0`
        - `robosuite 1.4.1`
      - frozen dataset download succeeded:
        - `low_dim_v141.hdf5`
   - first truthful failure:

```text
ModuleNotFoundError: No module named 'blind_diffusion.env'
```

   - interpretation:
     - this is now a repository packaging issue in the regression test path, not a backend or dependency-stack failure
     - the Antigravity target environment has effectively been reconstructed on Vast
   - repair:
     - add `src/blind_diffusion/env/__init__.py`
     - keep the workload, task, dataset family, and controller logic unchanged

## Planned retry 15

- preserve the same repaired-reference Vast workload
- preserve the same Python `3.10` / `robomimic 0.3.0` / `robosuite 1.4.1` / native MuJoCo `3.x` environment target
- preserve the frozen `low_dim_v141.hdf5` dataset family
- add only `src/blind_diffusion/env/__init__.py` to fix the repository package import
- submit with a new run identity
