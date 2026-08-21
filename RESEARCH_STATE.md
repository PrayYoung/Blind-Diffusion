# Blind-Diffusion Research State

Last updated: Thursday, August 21, 2026 UTC

## Frozen next experiment: Antigravity two-stage smoke

- Legacy 10k/100k world-model-only performance jobs are terminal; no job is active.
- The completed run was only `scripts/run_antigravity_two_stage_smoke.sh`, not a performance experiment.
- Frozen success criteria: native actions are unnormalized in both training datasets and WM checkpoint stats; WM and diffusion checkpoints are produced; diffusion RHC loads both and executes the first action from a DDIM-generated sequence.
- Smoke budget: 20 WM updates, 3 diffusion updates, one five-step episode. It cannot support any task-quality claim.
- RunPod submission `runpod-20260820T002144Z-aa8581a1` completed successfully with exit code 0, synchronized outputs, and runner-managed Pod termination. It produced both checkpoints and completed DDIM/RHC evaluation. The 0/1 success rate is non-diagnostic by the frozen smoke protocol.

## Antigravity-faithful single-seed result

- RunPod `runpod-20260820T003153Z-f47465bd` completed successfully: exit code 0, synced outputs, and runner-managed Pod termination.
- Seed 0 followed the frozen 20k WM / 15-epoch frozen-WM diffusion / DDIM-10 RHC protocol with native actions.
- WM loss declined from `2.6619` at step 0 to `1.0034` at step 19,900; best validation loss was `1.0243`.
- Diffusion mean epoch loss declined from `0.9946` to `0.6249`; the selected best checkpoint is epoch 14.
- All 20 Lift episodes had zero return and zero successes (average length 500). Classify the result as `STILL_NO_TASK_LEARNING`, not an execution failure and not a faithful reproduction of any Antigravity task-learning claim.
- Runtime was 52m17s end-to-end at the configured `$0.22/hr` rate (approximately `$0.19` before any provider billing rounding). No further seeds were launched.

## Offline learning diagnosis

- No new training, seed, or environment rollout was launched. On the saved held-out sequence split, normalized observation MSE was `0.00740`; the WM is not supported as the primary failure explanation.
- Raw KL averaged `0.5337`, and `98.6%` of positions were below the `1.0` free-bits threshold. Clamping therefore produced a KL term of `1.0033` and a recomputed total loss of `1.0203`; the apparent loss floor is mathematical, not unreconstructed observations.
- The trained diffusion policy failed offline under DDIM-10: generated native-action chunks reached absolute magnitude `4596.65` while demonstrations are bounded by `1.0`; held-out action MSE was `2.013e6` and mean correlation `0.0234`. This supports diffusion/sampling failure as the proximate cause of `0/20`.
- The dataset contains 200/200 reward-positive Lift demonstrations, 7-D OSC-pose actions, and a binary `{-1,+1}` gripper. The remaining conceptual implementation mismatch is temporal: WM training observes `(obs_t, action_t)`, while the pre-action policy/evaluator state observes `(obs_t, action_{t-1})`.

## Causal convention and sampler gate

- The causal convention is now explicit: `state[t] = posterior(obs[t], action[t-1])`; the policy chooses `action[t]` from that state, and the WM's imagined transition with `action[t]` produces `state[t+1]`. Old full-run checkpoints are incompatible with this repaired convention and require retraining.
- The old DDPM used the forward beta as sampling variance rather than the posterior variance. Both DDPM and DDIM lacked `x0` clipping, allowing invalid native actions to explode. Corrected DDIM-10 / DDPM samples from the old checkpoint are bounded but still weak: MSE `0.552` / `0.682`.
- A cheap, demo-disjoint offline-only gate (300 WM + 300 diffusion updates; 160 train / 40 held-out demos) failed: actions stayed bounded and MSE improved marginally to `0.548`, but mean correlation was only `0.026` against a required `0.10`. No simulator rollout was launched.

## UNet fresh-noise score gate

- Offline-only Vast run `vast-20260820T025127Z-95263-415ea0` completed successfully and its RTX 4090 instance was runner-terminated. It used one fixed native-range `7x8` target and `230-D` condition, 5,000 fresh-noise/timestep updates, and 4,000 held-out noise/timestep pairs; it loaded no dataset or environment.
- Production `UNet1D` and the supplied Antigravity UNet are byte-for-byte equivalent in this gate: both have 648,583 parameters and exactly the same outputs/metrics under the same seed. Fresh epsilon MSE was `0.02909`, but reconstructed-x0 MSE was `2186.47` because residual score error at high-noise timesteps is amplified by cosine `alpha_bar[99] = 2.43e-7`.
- A trainable timestep-affine score baseline (11,200 parameters; the exact score family for one fixed x0) reached fresh x0 MSE `3.48e-4`. This validates the synthetic objective and identifies the shared UNet terminal-noise score accuracy as the active blocker. It does not isolate the absence of encoder-decoder skips as the causal architectural reason.

## Offline direct-x0 gate: PASS

- The same UNet and full cosine schedule pass once the denoiser predicts x0 directly: one-target fresh x0 MSE is `1.57e-5` (velocity also passes at `0.00748`), versus epsilon's `2186.47`.
- A real-demo DDIM-100 overfit also passes: MSE `1.38e-6`, native action absmax `1.00050`, no final clipping.
- On a demonstration-disjoint causal offline gate (300 WM and 1,000 diffusion updates), bounded direct-x0 DDIM-10 reaches MSE `0.01840`, mean correlation `0.552`, and absmax `0.999995`. Corrected DDPM-100 is also stable: MSE `0.01364`, absmax `0.999995`.
- The bounded direct-x0 path is now implemented in production and independently revalidated there. RunPod `runpod-20260821T053611Z-24ada889` completed the frozen held-out gate (300 causal WM updates, frozen WM, 1,000 diffusion updates, no simulator): DDIM-10 MSE `0.0183888`, correlation `0.552311`, absmax `0.999995`; DDPM-100 MSE `0.0136476`, absmax `0.999995`. All preregistered thresholds passed without evaluator-side clipping.
- Classification: **offline correctness validated; simulator performance pending.** Direct bounded-x0 is intentionally a recovery-protocol change from Antigravity's historical epsilon objective. No full run has been launched.

## Current repository state

- Local repository is the source of truth.
- HEAD commit: `356dc691381feaf7c269801cb484caaa24fe486c`
- Branch at freeze time: `master`
- Active research branch: `codex/blind-diffusion-research-recovery`
- Working tree at freeze time: clean

## Active remote execution

- Latest RunPod runner ID: `runpod-20260818T070931Z-fbdcbfd3`
- Pod `w4jcgzkgu5sl8l` was automatically terminated after a `$0.22/hr` run.
- Passed: frozen dataset download, 9 regression tests, and 20-step world-model training; its checkpoint and metrics synchronized under `.runpod/runs/runpod-20260818T070931Z-fbdcbfd3/artifacts/`.
- Remaining engineering blocker: `robosuite 1.4.1` uses `MjData.qM`, which is incompatible with MuJoCo `3.x`. The RunPod Python `3.10` environment can use the compatible MuJoCo `2.3.7` wheel for the next unchanged smoke retry.
- Validated RunPod smoke: `runpod-20260818T072118Z-8a290821` completed at `$0.22/hr` on CUDA-enabled RTX 3090; its Pod was terminated. The frozen dataset download, all 9 regressions, 20-step training, checkpoint production, and two MPC episodes all passed. Result artifact: `.runpod/runs/runpod-20260818T072118Z-8a290821/artifacts/outputs/repaired_reference_smoke/result.json`.
- Critic note: the smoke's `0/2` success and zero return are expectedly underpowered after only 20 optimizer steps. They establish executability, not a control-quality or Antigravity-performance claim.

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
- Active remote backend prepared on Friday, August 14, 2026:
  - global `vast-run` via `vast-run run --project-root . "<command>"`
  - repo profile: `.vast/profile.toml`
  - remote setup script: `scripts/setup_vast_env.sh`
- Historical Kaggle workflow retained locally for provenance:
  - `/Users/peiyan1/Desktop/ai_agent_memory/scripts/kaggle_remote_smoke.py`
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

- `r1-repaired-branch-reference-run`
- Goal: establish the first executable repaired-branch reference run after classifying the frozen baseline as non-runnable
- Artifact target: `research/experiments/20260814_r1_repaired_reference_smoke`
- Most recent remote run:
  - run: `vast-20260818T050625Z-30412-e0253b`
  - terminal classification: engineering `WORKLOAD_FAILED`; instance `47998271` was destroyed
  - selected RTX 4090 host `121206`, South Korea, `$0.342/hr`
  - repository snapshot under test: `6732be7` (only adds `src/blind_diffusion/env/__init__.py`)
  - first failure: the local runner excludes any directory named `env` from its archive, omitting the tracked `src/blind_diffusion/env/` package and causing `ModuleNotFoundError`
  - secondary host-selection defect: the host supports CUDA `12.2` only, so the frozen PyTorch CUDA `12.4` runtime reports `cuda_available: false`
  - protocol unchanged: Python `3.10`, `robomimic 0.3.0`, `robosuite 1.4.1`, native MuJoCo `3.x`, and frozen `low_dim_v141.hdf5`

## Completed experiments

- `20260814_local_diagnostics`
  - Baseline classified as not executable locally as configured.
  - Two core Antigravity bug claims reproduced locally.
  - Local regression tests now pass for RSSM stepping and dataset lifecycle.
- `20260814_remote_smoke_r0`
  - Repo-local Kaggle runner prepared and packaged successfully.
  - Submission approval was obtained on Friday, August 14, 2026.
  - Retry chain:
    - `20260814_remote_smoke_r0`: wrong RoboMimic downloader CLI
    - `20260814_remote_smoke_r0_retry1`: download succeeded, dataset path normalization wrong
    - `20260814_remote_smoke_r0_retry2`: dataset path fixed, remote unittest invocation wrong
    - `20260814_remote_smoke_r0_retry3`: PASS
  - Validated remote smoke outcome on retry 3:
    - GPU: Tesla T4
    - CUDA available
    - RoboMimic low-dim dataset downloaded remotely
    - regression tests passed remotely
    - tiny world-model training smoke produced checkpoint and metrics
- `20260814_r0_baseline_smoke`
  - Source commit under test:
    - `356dc691381feaf7c269801cb484caaa24fe486c`
  - Retry chain:
    - `20260814_r0_baseline_smoke`: engineering failure, historical bundle omitted `requirements-kaggle.txt`
    - `20260814_r0_baseline_smoke_retry1`: baseline code failure in `src/blind_diffusion/data/robomimic_dataset.py`
  - Truthful baseline classification:
    - frozen baseline is not executable under the recovered Kaggle protocol
    - first blocking failure is the dataset constructor dereference of unopened `self._h5`

## Active retry chain

- `20260814_r1_repaired_reference_smoke`
  - first attempt reached remote training successfully
  - first failure is engineering-only:
    - `src/blind_diffusion/env/robomimic_env.py` passed `use_image_obj` to `robomimic`
    - Kaggle runtime expects `use_image_obs`
  - second attempt reached further:
    - dataset download passed
    - regression tests passed
    - world-model smoke training passed
  - second failure is still engineering-only:
    - Kaggle resolved `robomimic 0.3.0`
    - low-dimensional evaluation failed on `ModuleNotFoundError: No module named 'mujoco_py'`
  - third failure is still engineering-only:
    - Kaggle PyPI did not expose `robomimic >= 0.4.0`
    - pip-only version pinning cannot reach the modern stack we need
  - fourth failure is still engineering-only:
    - source installs succeeded with `robomimic 0.4.0` and `robosuite 1.5.1`
    - that stack downloaded `low_dim_v15.hdf5`, not the frozen-protocol `low_dim_v141.hdf5`
    - silently substituting datasets would violate the frozen comparison basis
  - fifth failure is still engineering-only:
    - the `v141`-aligned stack works for dataset download, tests, CUDA, and world-model smoke training
    - the only remaining blocker is `robomimic.env_robosuite` importing `mujoco_py` before rollout
  - sixth failure is still engineering-only:
    - direct `robosuite.make(...)` bypassed the `mujoco_py` wrapper path
    - the only remaining blocker is `robosuite 1.4.1` rejecting the `lite_physics` keyword
  - seventh failure is still engineering-only:
    - pinning MuJoCo back to `2.3.x` on Kaggle Python `3.12.13` fails during dependency installation because `mujoco 2.3.7` falls back to a source build and does not produce a usable wheel there
    - the direct `robosuite.make(...)` compatibility reconstruction path is now retired
  - eighth failure is still engineering-only:
    - the restored `create_env_from_metadata(...)` / `EnvRobosuite` path now launches on Kaggle
    - the only blocker is pip dependency re-resolution rejecting the intentional `mujoco 3.5.0` override after the explicit compatibility installs
  - redirected next step:
    - keep `robomimic.utils.env_utils.create_env_from_metadata(...)` / `EnvRobosuite`
    - keep the frozen `low_dim_v141.hdf5` dataset family
    - keep native DeepMind `mujoco 3.5.0`, `robosuite 1.4.1`, and `robomimic 0.3.0 --no-deps`
    - patch only the unconditional `import mujoco_py` import site with a local compatibility stub
    - add `--no-deps` to the follow-up requirements and editable installs so pip stops invalidating the intended compatibility stack

## Best result so far

- No validated performance result yet.
- Best evidence so far is a validated engineering result:
  - the repaired branch can execute a remote RoboMimic smoke end-to-end on Kaggle after infrastructure fixes
  - this is not yet a baseline scientific result

## Blockers

1. No baseline checkpoints or prior run artifacts have been found in the local repository yet.
2. Dataset paths and local RoboMimic assets are not yet confirmed.
3. Local environment still lacks `hydra-core` and `robomimic`.
4. Local machine has no CUDA device, so latency and control-quality claims still require remote execution.
5. No repaired-branch reference training or evaluation run has been recorded yet, so there is still no executable control-quality reference point.
6. The repaired-system Vast rollout now reaches the real workload, but the remote runtime still has remaining Python dependency gaps after the graphics repairs.

## Next action

1. Run the identical frozen smoke through the repo-local RunPod profile: Community RTX 3090, CUDA `>=12.4`, and runner-managed cleanup.
2. Keep the environment target, dataset family, task, and repaired-controller logic frozen.
3. If it clears regression tests and reaches training plus low-dimensional rollout, freeze that as the working reproduction environment.
