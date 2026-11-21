# Validated Lift recovery result

The public headline is **19/20 clean and 18/20 permanently blind task successes**
for the OT-path conditional flow policy. Counts were recomputed from downloaded
per-episode metrics, and all 20 paired initial observation vectors were checked
for exact equality. The curated [machine-readable summary](results/lift_flow_matching.json)
contains no checkpoints, credentials, host identifiers, or private paths.

## Protocol

- RoboMimic Lift PH, `low_dim_v141.hdf5`; 200 demonstrations split into 190
  training and 10 validation demonstrations, seed 0, shared between both stages.
- Observation normalization uses training demonstrations only; actions remain
  in their native range.
- World model: `train_world_model_recovery`, 4,000 updates, batch 128. Minimum
  total held-out objective selected update 2,000.
- Policy: `train_flow_matching_recovery`, 100 epochs, batch 128; EMA checkpoint
  selected by held-out velocity loss with four fixed Monte Carlo draws. Epoch
  99 (zero-based) was selected, held-out velocity loss 0.11604.
- Execution: Euler-4, action horizon 8, replan each environment step, execute
  the first action. Deterministic RSSM mean beliefs and executed-action history.
- Evaluation seeds 0–19, 500 steps per episode, same checkpoint and initial
  observations for both conditions. Clean receives observations throughout.
  Blind receives observations at steps 0–4 and uses the RSSM prior thereafter.
- Success is the environment task predicate becoming true at any point during
  an episode. It does not require the object to remain lifted until step 500.
- Predeclared screen: five paired seeds, expanding to 20 if at least three
  clean episodes succeeded. All five clean episodes succeeded; all subsequent
  results, including failures, were retained.

## Every outcome

| Paired result | Seeds | Count |
| --- | --- | ---: |
| Both succeed | 1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 | 17 |
| Clean succeeds, blind fails | 0, 2 | 2 |
| Blind succeeds, clean fails | 7 | 1 |
| Both fail | — | 0 |

The five-percentage-point difference is descriptive, not evidence of a
statistically established performance gap between conditions.

## Engineering checks and diagnostic limits

Held-out normalized world-model reconstruction MSE was 0.01379; mean prior MSE
over offsets 1–20 was 0.10165. A separate final-offset-20 audit measured 0.19836,
narrowly passing its predeclared 0.20 threshold (persistence reference 1.35676).
These expert-action-conditioned metrics are not guarantees of closed-loop
prediction accuracy.

A selected seed-0 replay reproduced clean success and blind failure. Blind
closed its gripper seven steps earlier, reached a minimum end-effector/object
distance of 3.49 cm versus 1.18 cm clean, and had approximately 3.72 cm of decoded
end-effector position error near closest approach. This supports investigating
prediction precision and grasp timing; passive decoder errors do not prove
the cause of failed control. Large late clean latent excursions occurred after
the task had already succeeded, so latent norm growth alone is not a sufficient
failure explanation.

## What this result supports

The versioned recovery pipeline learns clean and blind simulated Lift control.
It does not isolate the individual contributions of repaired data handling,
causal alignment, optimization, architecture, or policy objective. Historical
failed runs used different configurations and are not a matched baseline for
an improvement percentage.

This is one training seed, one task, and 20 paired evaluation seeds. No Can,
image-control, latency, real-robot, or baseline-superiority claim is made here.
Claims from other experiment reports require their own raw metrics and protocol
audit before inclusion.

## Reproduce and compare

Use the [tested installation and commands](setup.md) to rerun the experiment.
The public JSON records the initial 20-pair result; it does not generate new
rollouts. A larger matched baseline campaign is in progress, with separate
protocols and outcome records. No incomplete cells or external screenshot
claims are included in the headline table.
