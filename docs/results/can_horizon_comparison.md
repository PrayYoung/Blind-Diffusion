# Can: fixed checkpoints, longer-horizon comparison

30 paired seeds 0–29; one training seed; all deadlines scored from each same trajectory. No retraining.

## Success by step 115 (secondary)

| Method | D=0 | D=5 | D=10 | D=15 | D=20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cfm4 | 8/30 (26.7%) | 6/30 (20.0%) | 7/30 (23.3%) | 8/30 (26.7%) | 10/30 (33.3%) |
| ddim10 | 10/30 (33.3%) | 10/30 (33.3%) | 9/30 (30.0%) | 10/30 (33.3%) | 9/30 (30.0%) |
| open_loop | 9/30 (30.0%) | 2/30 (6.7%) | 2/30 (6.7%) | 0/30 (0.0%) | 0/30 (0.0%) |
| bc | 0/30 (0.0%) | 0/30 (0.0%) | 0/30 (0.0%) | 0/30 (0.0%) | 0/30 (0.0%) |

## Success by step 250 (secondary)

| Method | D=0 | D=5 | D=10 | D=15 | D=20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cfm4 | 21/30 (70.0%) | 26/30 (86.7%) | 26/30 (86.7%) | 26/30 (86.7%) | 21/30 (70.0%) |
| ddim10 | 24/30 (80.0%) | 26/30 (86.7%) | 22/30 (73.3%) | 25/30 (83.3%) | 22/30 (73.3%) |
| open_loop | 24/30 (80.0%) | 7/30 (23.3%) | 6/30 (20.0%) | 0/30 (0.0%) | 0/30 (0.0%) |
| bc | 5/30 (16.7%) | 0/30 (0.0%) | 0/30 (0.0%) | 0/30 (0.0%) | 0/30 (0.0%) |

## Success by step 300 (primary)

| Method | D=0 | D=5 | D=10 | D=15 | D=20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cfm4 | 21/30 (70.0%) | 26/30 (86.7%) | 26/30 (86.7%) | 26/30 (86.7%) | 21/30 (70.0%) |
| ddim10 | 24/30 (80.0%) | 26/30 (86.7%) | 22/30 (73.3%) | 25/30 (83.3%) | 22/30 (73.3%) |
| open_loop | 24/30 (80.0%) | 7/30 (23.3%) | 6/30 (20.0%) | 0/30 (0.0%) | 0/30 (0.0%) |
| bc | 5/30 (16.7%) | 0/30 (0.0%) | 0/30 (0.0%) | 0/30 (0.0%) | 0/30 (0.0%) |

Archived v1 prefix checks: 600; mismatches: 5.
The original 115-step benchmark is preserved separately. This run does not reproduce the external reference's different seeds/checkpoints.


## Interpretation and scope

The fixed-checkpoint deadline comparison is complete: 600 episodes, four
methods, five blackout conditions, 30 paired evaluation seeds, one training
seed. Blackout starts after five visible observations and repeats every 20
steps; D=20 is permanent blackout. The primary deadline is 300 steps. All
300-step successes already occurred by step 250. This is a versioned extension,
not a replacement for the original 115-step Can benchmark.

DDIM uses bounded direct-x0 prediction, not epsilon prediction. CFM-4 and
DDIM-10 replan each step. The zero-input open-loop baseline uses the same DDIM
checkpoint, executes eight-action chunks, and updates the RSSM posterior from
zero-filled normalized observations during blackout; it does not take the
prior-only imputation branch. Reactive BC has no recurrent state. Its weak
clean score limits broader conclusions about behavior cloning.

CFM-4 uses four model evaluations per plan versus ten for DDIM-10. These counts
do not establish a measured runtime speedup. The observed success rates do not
establish CFM superiority, statistical equivalence, or reproduction of the
external 90%/70% result with different checkpoints and seeds. The open-loop
comparison changes both observation handling and action execution frequency;
a same-checkpoint, same-replanning ablation is still needed to isolate the
contribution of prior-only imputation.

## Validation and reproducibility

All 600 raw episode records passed grid, checkpoint-hash, same-trajectory
deadline-count and exact within-host starting-state/observation pairing checks.
All 600 initial physical-state hashes match the archived benchmark. Five
115-step prefixes differ across hosts: two success flags and three success
timings. The original benchmark remains unchanged. The full repository test
suite passed (51 tests in the research workspace).

The selected checkpoint hashes and complete protocol are recorded in
[`can_horizon_protocol_v2.json`](../../scripts/can_horizon_protocol_v2.json).
Revision 1 audits small cross-host observation roundoff while retaining exact
physical states and exact same-host pairing. Evaluation used CPU inference,
Python 3.10, PyTorch 2.4, MuJoCo 2.3.7 and RoboSuite 1.4.1. No retraining was
performed for the deadline extension.

The portable entrypoints are `scripts/run_matched_campaign.py` for training
and the original comparison, `scripts/run_can_horizon_eval.py --cpu-only` for
the extension, and `scripts/collect_can_horizons.py` for its tables. Use each
script's `--help`. The extension requires the selected checkpoints and archived
initial-state/reference records; these private research artifacts are not
bundled. The summary alone is not a self-contained reproduction package.
