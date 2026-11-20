# Recovery on an existing Colab runtime

The `scripts/colab_*.py` files run through `colab exec -s SESSION -f SCRIPT`.
They use the already connected runtime: they do not allocate, restart, or stop
it, mount Drive, or alter the notebook kernel's installed packages. The recovery
campaign uses a separate Python 3.10 environment under
`/content/blind_diffusion_ee1bb70_20261120`.

These are versioned campaign scripts. Their task directory, payload name, and
experiment protocol are explicit rather than inferred from mutable session
state. To start a different campaign, version those names and its protocol;
do not delete guards to repeat completed or negative experiments.

1. Run `scripts/colab_preflight.py` to inspect CUDA, free memory, and active GPU
   processes. Reuse the session only when its GPU is available for this job.
2. Create a code-only archive named `blind_diffusion_payload_ee1bb70.tar.gz`
   containing `src/`, `README.md`, `pyproject.toml`, `requirements-vast.txt`,
   `scripts/cuda_preflight.py`, and `scripts/run_recovery_v2_smoke.sh`.
   Exclude `__pycache__`, credentials, local state, datasets, and checkpoints.
   Upload it to `/content/` using the Colab CLI.
3. Execute `scripts/colab_recovery_smoke.py`. It creates the isolated environment,
   downloads public Lift demonstrations, and runs a bounded two-stage CUDA smoke.
   Retrieve and inspect its archive before full training. A smoke's task success
   is not a performance result.
4. Execute `scripts/colab_train_world_model.py` once. Its state file prevents
   duplicate submission. Retrieve its checkpoint and logs.
5. Upload `scripts/audit_prior_predictions.py` to the task's `scripts/` directory,
   then execute `scripts/colab_audit_world_model.py`. The fixed gates are held-out
   reconstruction MSE ≤0.04, average prior MSE ≤0.20, and offset-20 MSE ≤0.20.
6. If all gates pass, execute `scripts/colab_train_flow_policy.py` once. It trains
   the fixed 100-epoch policy and selects EMA weights by held-out velocity loss.
7. Upload `scripts/audit_flow_imitation.py`, then execute
   `scripts/colab_audit_flow_policy.py` for a paired Euler-4/Euler-32 diagnostic.
   This diagnostic does not change the control benchmark.
8. Execute `scripts/colab_evaluate_flow_policy.py` for five paired clean and
   permanent-blackout episodes. It verifies equal initial observations and adds
   seeds 5–19 only when at least three of the first five clean episodes succeed.
   Report every outcome, including failures.

Inspect explicit remote state and expected output files: the Colab CLI's own
exit code alone may not distinguish a Python workload exception. Download
archives and audits locally; keep them out of source-control changes. The
existing Colab runtime remains owned by its user after the job completes.

`scripts/colab_continue_smoke.py` is a recovery entry point for a recorded,
diagnosed setup-only failure. It must not be used to overwrite a running or
completed smoke, or to retry an unfavorable scientific result.
