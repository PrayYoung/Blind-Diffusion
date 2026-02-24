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

## Project layout

- `configs/`: task, model, planner, and run configs
- `scripts/`: CLI entry points
- `src/blind_diffusion/`: library code
- `runs/`: outputs (checkpoints, logs, metrics)

## Notes

- Milestone 1 uses low-dim observations only (no images).
- Collision/constraint violation is defined in `src/blind_diffusion/planner/cost.py`.
