# Tested environment and data setup

The verified Lift run used Linux, Python 3.10.22, PyTorch 2.4.0 + CUDA 12.4,
MuJoCo 2.3.7, RoboSuite 1.4.1, RoboMimic 0.3.0, and an NVIDIA T4.

## Install

Use a clean Python 3.10 environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-vast.txt
python -m pip install mujoco==2.3.7 robosuite==1.4.1
python -m pip install --no-deps robomimic==0.3.0
python -m pip install -e .
python -m pip install pytest
export ROBO_DATA="$PWD/robomimic_datasets"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MPLBACKEND=Agg
```

MuJoCo 2.3.7 matches the controller API used by RoboSuite 1.4.1 in the tested
environment. The package is installed editable so Hydra can find the source
configuration files.

## Download Lift

```bash
python -m robomimic.scripts.download_datasets \
  --tasks lift --dataset_types ph --hdf5_types low_dim --download_dir "$ROBO_DATA"
cp "$ROBO_DATA/lift/ph/low_dim_v141.hdf5" "$ROBO_DATA/lift/low_dim_v141.hdf5"
```

The configured task path is `lift/low_dim_v141.hdf5` relative to `ROBO_DATA`.

## Train and evaluate

```bash
python -m blind_diffusion.train.train_world_model --config-name train_world_model_recovery num_workers=0
python -m blind_diffusion.train.train_diffusion --config-name train_flow_matching_recovery num_workers=0
python -m blind_diffusion.eval.evaluate_diffusion --config-name eval_flow_matching \
  exp_name=eval_clean episodes=20 seed=0 max_steps=500 blackout_duration=0
python -m blind_diffusion.eval.evaluate_diffusion --config-name eval_flow_matching \
  exp_name=eval_blind episodes=20 seed=0 max_steps=500 \
  warmup_steps=5 blackout_period=20 blackout_duration=20
```

Default training outputs are `runs/wm_lift_recovery_v2/` and
`runs/flow_lift_recovery_v2/`. Evaluation records are written to
`runs/<exp_name>/diffusion_metrics.json`, including per-episode seeds, outcomes,
and initial observations. Use the same checkpoints and seeds in both conditions.

`blackout_duration=20` with `blackout_period=20` means no observations return
after the five-step warmup. Values 5, 10, and 15 instead create periodic
blackout; those conditions are not part of the initial 20-pair headline result.

## Engineering checks

```bash
python -m pytest -q
python scripts/cuda_preflight.py
bash scripts/run_recovery_v2_smoke.sh
```

The CUDA checks require a suitable NVIDIA GPU. The bounded smoke trains for
only 20 world-model and 10 policy updates, then executes short clean/blind
episodes. It verifies the pipeline, not task quality.
