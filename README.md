# [BD] blind-diffusion

Tiny RSSM + MPC for RoboMimic low-dim control.

**License:** MIT (see `LICENSE`)

Milestone 1: Train a small RSSM world model on RoboMimic low-dimensional observations, then run receding-horizon MPC (CEM or MPPI) in the learned latent dynamics. Report success rate and collision/constraint-violation rate, plus an open-loop baseline.

## Quickstart

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
export PYTHONPATH=./src
```

If you need RoboMimic env evaluation on Linux:
```bash
uv pip install -e ".[robomimic]"
```

## RoboMimic data

1) Install `robomimic` and its dependencies (see RoboMimic docs).
2) Download low-dim datasets (e.g., `lift`):
   - `https://robomimic.github.io/docs/datasets/overview.html`
3) Set `ROBO_DATA` to the folder containing the `.hdf5` file.

Example:
```bash
export ROBO_DATA=/path/to/robomimic/datasets
```

## Train world model

```bash
python scripts/train_wm.py \
  --config configs/train_world_model.yaml \
  task=lift
```

Outputs:
- `runs/<exp_name>/checkpoints/*.pt`
- `runs/<exp_name>/logs.jsonl`
- `runs/<exp_name>/metrics.json`

## Train open-loop baseline (BC)

```bash
python scripts/eval_open_loop.py \
  --config configs/eval_mpc.yaml \
  task=lift \
  eval.mode=bc_train
```

Evaluate BC:
```bash
python scripts/eval_open_loop.py \
  --config configs/eval_mpc.yaml \
  task=lift \
  eval.mode=bc_eval
```

## Evaluate MPC

```bash
python scripts/eval_mpc.py \
  --config configs/eval_mpc.yaml \
  task=lift
```

## Milestone 2: Diffusion policy

Train diffusion policy (conditioned on frozen RSSM belief):
```bash
python scripts/train_diffusion.py \
  --config configs/train_diffusion.yaml \
  task=lift
```

Evaluate diffusion policy (receding horizon control):
```bash
python scripts/eval_diffusion.py \
  --config configs/eval_diffusion.yaml \
  task=lift
```

Open-loop diffusion baseline:
```bash
python scripts/eval_diffusion.py \
  --config configs/eval_diffusion.yaml \
  task=lift \
  eval.mode=open_loop
```

## Milestone 3: Image-based world model + diffusion

Train image-based world model:
```bash
python scripts/train_wm_image.py \
  --config configs/train_world_model_image.yaml \
  task=lift
```

Train image-conditioned diffusion:
```bash
python scripts/train_diffusion_image.py \
  --config configs/train_diffusion_image.yaml \
  task=lift
```

Evaluate image-conditioned diffusion:
```bash
python scripts/eval_image_policy.py \
  --config configs/eval_image.yaml \
  task=lift
```

Open-loop image diffusion baseline:
```bash
python scripts/eval_image_policy.py \
  --config configs/eval_image.yaml \
  task=lift \
  eval.mode=open_loop
```

Train image BC baseline:
```bash
python scripts/train_bc_image.py \
  --config configs/train_bc_image.yaml \
  task=lift
```

Evaluate image BC baseline:
```bash
python scripts/eval_image_policy.py \
  --config configs/eval_image.yaml \
  task=lift \
  eval.mode=bc_eval
```

## Project layout

- `configs/`: task, model, planner, and run configs
- `scripts/`: CLI entry points
- `src/blind_diffusion/`: library code
- `runs/`: outputs (checkpoints, logs, metrics)

## Notes

- Milestone 1 uses low-dim observations only (no images).
- Collision/constraint violation is defined in `src/blind_diffusion/planner/cost.py`.
