# Blind-Diffusion

Blind-Diffusion explores whether a recurrent latent world model can support
diffusion-based robotic control under partial observation. It combines an RSSM
belief state with a diffusion action policy and receding-horizon execution.

## Highlights

- Causal recurrent RSSM beliefs: `belief[t] = posterior(obs[t], action[t-1])`.
- Bounded direct-`x0` diffusion for native low-dimensional actions.
- DDIM and DDPM action-sequence sampling for receding-horizon control.
- RoboMimic / RoboSuite integration with dataset controller metadata preserved.
- Offline imitation is validated; the closed-loop Lift limitation is characterized below.

## Results

| Evaluation | Result |
| --- | ---: |
| Held-out DDIM-10 imitation MSE | 0.0184 |
| Held-out mean action correlation | 0.552 |
| Action range | intrinsically bounded to approximately `[-1, 1]` |
| Clean Lift closed-loop success | 0 / 20 |
| Clean Lift average return | 0.0 |

**Key finding:** Offline belief-conditioned imitation is accurate and bounded,
but it does not translate into successful closed-loop Lift control; our
diagnostics indicate that policy-induced observation shift drives the recurrent
latent state outside the expert distribution.

## Closed-loop failure analysis

Our diagnostics identify policy-induced observation distribution shift and
recurrent latent drift as the dominant observed failure mode in the tested
setting:

```text
policy deviation
  → observation distribution shift
  → encoder / RSSM posterior-mean drift
  → latent state leaves the expert distribution
  → policy receives unfamiliar conditioning
```

![Closed-loop latent drift](assets/latent_drift.png)

*Recorded closed-loop trajectories and the validated teacher-forced reference.*
The sampled latent norm reaches `||z||₂ ≈ 71.75`, compared with a
teacher-forced demonstration reference maximum of `≈ 11.19`. A safe deployable
remedy has not yet been established.

## Recovered implementation

- Safe lazy RoboMimic HDF5 lifecycle for dataset construction and workers.
- Persistent causal RSSM state propagation and pre-action temporal alignment.
- Direct-`x0` diffusion with an intrinsic `tanh` action bound.
- Matching DDIM/DDPM production sampling equations and focused sampler tests.
- Preservation of the dataset's 7-D `OSC_POSE` controller configuration during evaluation.

## Current status

The repository is executable and its offline diffusion/imitation behavior is
validated. It is **not** an end-to-end successful Lift controller. Future work
will likely require recovery supervision or another robustness mechanism for
policy-induced out-of-distribution states.

## Quick start

Python 3.10+ and PyTorch are required. Install the package with the RoboMimic
extras in a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[robomimic]"
export PYTHONPATH="$PWD/src"
```

Download the RoboMimic low-dimensional dataset and point `ROBO_DATA` to its
parent directory:

```bash
python -m robomimic.scripts.download_datasets \
  --tasks lift --dataset_types ph --hdf5_types low_dim \
  --download_dir ./robomimic_datasets
export ROBO_DATA="$PWD/robomimic_datasets"
```

Run the focused regression tests:

```bash
python -m unittest discover -s tests
```

Then train a low-dimensional world model, train the diffusion policy, and
evaluate with receding-horizon control:

```bash
python scripts/train_wm.py task=lift
python scripts/train_diffusion.py task=lift
python scripts/eval_diffusion.py task=lift eval.mode=rhc
```

The default Hydra configs live in `src/blind_diffusion/configs/`.

## Repository layout

```text
src/blind_diffusion/  models, data, training, evaluation, and configs
scripts/              public training and evaluation entrypoints
tests/                focused regression tests
assets/               public result figure
```

## License

MIT. See [LICENSE](LICENSE).
