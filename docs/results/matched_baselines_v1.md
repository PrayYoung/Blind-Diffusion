# Original matched comparison — protocol v1

For the current 300-step Can extension, see [Can horizon comparison](can_horizon_comparison.md).
The original 115-step Can results below remain unchanged.

One training seed. Success at any time within the task horizon. Intervals: Wilson 95%.

## Lift (N=50, max steps=500)

| Method | D=0 | D=5 | D=10 | D=15 | D=20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OT-path CFM, Euler4 + prior | 49/50 (98.0%; 89.5–99.6) | 50/50 (100.0%; 92.9–100.0) | 50/50 (100.0%; 92.9–100.0) | 48/50 (96.0%; 86.5–98.9) | 45/50 (90.0%; 78.6–95.7) |
| OT-path CFM, Euler2 + prior | 45/50 (90.0%; 78.6–95.7) | 47/50 (94.0%; 83.8–97.9) | 47/50 (94.0%; 83.8–97.9) | 46/50 (92.0%; 81.2–96.8) | 44/50 (88.0%; 76.2–94.4) |
| Direct-x0 DDIM10 + prior | 49/50 (98.0%; 89.5–99.6) | 49/50 (98.0%; 89.5–99.6) | 47/50 (94.0%; 83.8–97.9) | 48/50 (96.0%; 86.5–98.9) | 45/50 (90.0%; 78.6–95.7) |
| DDIM10, H8 chunks, zero-filled observations | 50/50 (100.0%; 92.9–100.0) | 50/50 (100.0%; 92.9–100.0) | 34/50 (68.0%; 54.2–79.2) | 11/50 (22.0%; 12.8–35.2) | 0/50 (0.0%; 0.0–7.1) |
| Reactive BC, zero-filled observations | 44/50 (88.0%; 76.2–94.4) | 49/50 (98.0%; 89.5–99.6) | 24/50 (48.0%; 34.8–61.5) | 0/50 (0.0%; 0.0–7.1) | 0/50 (0.0%; 0.0–7.1) |
| Latent CEM, learned sparse reward | 0/50 (0.0%; 0.0–7.1) | 0/50 (0.0%; 0.0–7.1) | 0/50 (0.0%; 0.0–7.1) | 0/50 (0.0%; 0.0–7.1) | 0/50 (0.0%; 0.0–7.1) |

Latency: batch-one observation-to-action computation, no simulation; CPU uses one thread.
GPU: NVIDIA RTX 4000 Ada Generation; PyTorch: 2.4.0+cu124.

| Method | CPU visible median / p95 (ms) | GPU visible median / p95 (ms) |
| --- | ---: | ---: |
| OT-path CFM, Euler4 + prior | 11.47 / 11.80 | 10.76 / 10.90 |
| OT-path CFM, Euler2 + prior | 6.00 / 6.21 | 5.69 / 5.81 |
| Direct-x0 DDIM10 + prior | 29.73 / 30.55 | 29.37 / 30.08 |
| DDIM10, H8 chunks, zero-filled observations | 30.60 / 33.59 | 29.53 / 29.89 |
| Reactive BC, zero-filled observations | 0.09 / 0.10 | 0.24 / 0.25 |
| Latent CEM, learned sparse reward | 141.42 / 142.23 | 16.70 / 16.89 |

Open-loop latency above is per plan query; its measured chunk-amortized per-step cost and blind-path timings are retained in the latency JSON.

## Can (N=30, max steps=115)

| Method | D=0 | D=5 | D=10 | D=15 | D=20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OT-path CFM, Euler4 + prior | 7/30 (23.3%; 11.8–40.9) | 6/30 (20.0%; 9.5–37.3) | 8/30 (26.7%; 14.2–44.4) | 8/30 (26.7%; 14.2–44.4) | 10/30 (33.3%; 19.2–51.2) |
| Direct-x0 DDIM10 + prior | 10/30 (33.3%; 19.2–51.2) | 10/30 (33.3%; 19.2–51.2) | 9/30 (30.0%; 16.7–47.9) | 10/30 (33.3%; 19.2–51.2) | 9/30 (30.0%; 16.7–47.9) |
| DDIM10, H8 chunks, zero-filled observations | 9/30 (30.0%; 16.7–47.9) | 2/30 (6.7%; 1.8–21.3) | 2/30 (6.7%; 1.8–21.3) | 0/30 (0.0%; 0.0–11.4) | 0/30 (0.0%; 0.0–11.4) |
| Reactive BC, zero-filled observations | 0/30 (0.0%; 0.0–11.4) | 0/30 (0.0%; 0.0–11.4) | 0/30 (0.0%; 0.0–11.4) | 0/30 (0.0%; 0.0–11.4) | 0/30 (0.0%; 0.0–11.4) |

Latency: batch-one observation-to-action computation, no simulation; CPU uses one thread.
GPU: NVIDIA RTX 4000 Ada Generation; PyTorch: 2.4.0+cu124.

| Method | CPU visible median / p95 (ms) | GPU visible median / p95 (ms) |
| --- | ---: | ---: |
| OT-path CFM, Euler4 + prior | 11.19 / 11.49 | 10.16 / 10.22 |
| Direct-x0 DDIM10 + prior | 29.45 / 30.71 | 28.30 / 29.57 |
| DDIM10, H8 chunks, zero-filled observations | 29.41 / 30.78 | 28.52 / 29.67 |
| Reactive BC, zero-filled observations | 0.09 / 0.10 | 0.23 / 0.26 |

Open-loop latency above is per plan query; its measured chunk-amortized per-step cost and blind-path timings are retained in the latency JSON.

D is the number of blind steps per 20-step cycle after five visible observations; D=20 is permanent blackout.
These are explicitly defined implementation baselines, not an audited reproduction of external results.
