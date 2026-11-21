# Blind-Diffusion

**Causal world-model beliefs and flow-matched action plans for control when observations stop.**

Blind-Diffusion learns from offline robot demonstrations. A recurrent
state-space model (RSSM) remembers the scene; a conditional flow policy turns
that belief into actions, including during observation blackout.

## Verified Lift result

| Policy | Clean observations | Permanent blackout after 5 observations |
| --- | ---: | ---: |
| OT-path CFM, 4 Euler steps + RSSM prior | **95% (19/20)** | **90% (18/20)** |

RoboMimic Lift, low-dimensional observations, **one training seed and 20 paired
evaluation seeds**. The same checkpoint is used in both conditions. Success
means the environment task predicate becomes true within 500 steps; it does
not require a sustained hold until the final step.

**What this establishes:** this configuration learns clean and blind simulated
Lift control. A larger matched comparison with BC, DDIM, open-loop execution,
and latent MPC is in progress. It is not yet a verified baseline-superiority,
Can, vision, real-robot, or latency result.

[Recorded counts and per-seed outcomes](docs/results/lift_flow_matching.json)
· [Full protocol and evidence limits](docs/validated_results.md)

## How it works

1. **Observe:** update the RSSM posterior from the current observation and the
   previous executed action: `belief[t] = posterior(obs[t], action[t-1])`.
2. **Remember:** during blackout, use the action-conditioned prior instead:
   `belief[t] = prior(belief[t-1], action[t-1])`. No new observations update it.
3. **Act:** generate an 8-action chunk using four Euler steps, execute its first
   action, and replan at the next environment step.

The flow policy uses Gaussian optimal-transport conditional paths with
independent Gaussian/action pairs. It does **not** perform minibatch OT
assignment. Velocity and intermediate ODE states are unbounded; completed
native actions clamp once to `[-1, 1]`.

## Evidence checks

- **Held-out demonstrations:** both training stages share a seeded 190/10
  episode split; normalization uses training demonstrations only.
- **Causal conditioning:** regression tests perturb future actions and verify
  that earlier decision beliefs remain unchanged.
- **Paired starts:** all 19 initial observation coordinates match exactly for
  every clean/blind pair in the reported run.
- **Sensor isolation:** after observations at steps 0–4, the blind controller
  uses only its remembered state and executed-action history. True observations
  may be logged for diagnostics but do not update its belief.

## Run it

Use the [tested installation and dataset setup](docs/setup.md) first
(Python 3.10, PyTorch 2.4 + CUDA 12.4, MuJoCo 2.3.7).

```bash
python -m blind_diffusion.train.train_world_model --config-name train_world_model_recovery num_workers=0
python -m blind_diffusion.train.train_diffusion --config-name train_flow_matching_recovery num_workers=0
python -m blind_diffusion.eval.evaluate_diffusion --config-name eval_flow_matching exp_name=eval_clean episodes=20 seed=0
python -m blind_diffusion.eval.evaluate_diffusion --config-name eval_flow_matching exp_name=eval_blind episodes=20 seed=0 blackout_duration=20
```

Run the regression suite with `python -m pytest -q`. A short CUDA execution
check is available as `bash scripts/run_recovery_v2_smoke.sh`; it cannot establish
task performance with its small training budget.

<details>
<summary>Training protocol, limitations, and next comparisons</summary>

The reported run used 4,000 world-model updates and 100 policy epochs, batch
128, deterministic RSSM mean beliefs, padding masks, multi-step prior losses,
blackout-conditioned policy training, and EMA checkpoint selection by held-out
velocity loss. Evaluation used seeds 0–19 and Euler-4 receding-horizon control.
Blackout is permanent after the first five observations.

The data-handling, optimization, architecture, and objective changes were
introduced together. These results do not isolate which individual repair
caused historical failures. Historical failed runs are not a matched baseline
for an improvement percentage.

Next evidence to establish: additional training seeds, the complete matched
blackout comparison, Can and other tasks, and partial-modality or image-based
control. Legacy bounded direct-`x0` DDIM/DDPM, latent MPC, and image entrypoints
remain available for experiments; their availability is not a validated result.

Raw checkpoints, datasets, private execution logs, and personal infrastructure
state are not distributed in this repository.

</details>

Source: [`src/blind_diffusion/`](src/blind_diffusion/). License: [MIT](LICENSE).
