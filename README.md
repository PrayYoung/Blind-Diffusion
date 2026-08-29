# Blind-Diffusion

Blind-Diffusion is a compact research implementation of belief-conditioned
diffusion control for RoboMimic manipulation. It combines a recurrent state
space model (RSSM), which turns observations into a causal latent belief, with
a diffusion model that proposes short action sequences from that belief.

The project is useful as a readable reference for a difficult interface in
robot learning: keeping world-model training, action conditioning, diffusion
training, and receding-horizon evaluation on the same causal convention.

## What is implemented

- Low-dimensional and image-conditioned RSSM world models.
- Diffusion action-sequence generation conditioned on RSSM beliefs.
- DDIM and DDPM sampling for native RoboMimic actions.
- Receding-horizon and open-loop evaluation entrypoints.
- RoboMimic Lift / Can configurations and focused regression tests.

## Corrected implementation

This public version includes recovered correctness fixes for recurrent state
propagation, HDF5 dataset lifecycle, environment metadata handling, and causal
pre-action belief alignment. The low-dimensional diffusion path uses direct
`x0` prediction with an intrinsic `tanh` action bound; its DDIM and DDPM
samplers share the corresponding production equations.

## Validation status

Focused regression tests validate the causal belief convention, native action
handling, bounded direct-`x0` sampling, DDIM/DDPM behavior, RSSM transitions,
dataset construction, and environment metadata.

Offline diffusion correctness and imitation behavior have been validated for
the corrected implementation. **Closed-loop Lift task success is not yet
established.** This repository is a research codebase, not a claim of a
solved end-to-end manipulation benchmark.

## Setup

Python 3.10+ and PyTorch are required. Install the package and the RoboMimic
extras in a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[robomimic]"
export PYTHONPATH="$PWD/src"
```

Download the RoboMimic low-dimensional dataset and point `ROBO_DATA` at its
parent directory:

```bash
python -m robomimic.scripts.download_datasets \
  --tasks lift --dataset_types ph --hdf5_types low_dim \
  --download_dir ./robomimic_datasets
export ROBO_DATA="$PWD/robomimic_datasets"
```

## Minimal reproduction

Run the focused tests first:

```bash
python -m unittest discover -s tests
```

Then train a low-dimensional world model, train the diffusion policy, and
evaluate it with receding-horizon control:

```bash
python scripts/train_wm.py task=lift
python scripts/train_diffusion.py task=lift
python scripts/eval_diffusion.py task=lift eval.mode=rhc
```

The default configs are in `src/blind_diffusion/configs/`. Override them with
Hydra arguments as needed, for example `task=can` or
`diffusion=diffusion_full`.

## Repository layout

```text
src/blind_diffusion/  core models, training, evaluation, and configs
scripts/              public training and evaluation entrypoints
tests/                focused regression tests
```

## License

MIT. See [LICENSE](LICENSE).
