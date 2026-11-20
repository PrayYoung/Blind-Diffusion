"""Matched one-action-chunk DDIM/OT-path CFM implementation gate.

No world-model conditioning or simulator: passing isolates the generative
objective/sampler plumbing and does not establish imitation/control success.
"""
import argparse
import json
from pathlib import Path
import time

import h5py
import torch

from blind_diffusion.diffusion.policy import build_action_policy
from blind_diffusion.utils.ema import ModelEMA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--base-ch", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    with h5py.File(args.dataset, "r") as handle:
        name = sorted(handle["data"].keys())[0]
        target = torch.tensor(handle[f"data/{name}/actions"][:8], device=device, dtype=torch.float32).T[None]
    torch.manual_seed(17)
    condition = torch.randn(1, 230, device=device)
    result = {"kind": "single_target_engineering_gate", "demo": name, "anchor": 0,
              "steps": args.steps, "base_ch": args.base_ch, "device": str(device), "results": {}}
    for kind in ("diffusion", "flow_matching"):
        torch.manual_seed(19)
        config = {"policy_type": kind, "flow": {"sigma_min": 1e-4, "time_scale": 1000.},
            "diffusion": {"architecture": "skip_v2", "base_ch": args.base_ch, "horizon": 8,
                          "timesteps": 100, "schedule": "cosine", "prediction_type": "x0"}}
        policy = build_action_policy(config, 7, 230).to(device)
        ema = ModelEMA(policy)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
        started = time.monotonic()
        for step in range(args.steps):
            loss = policy.loss(target.expand(args.batch_size, -1, -1), condition.expand(args.batch_size, -1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.)
            optimizer.step()
            scheduler.step()
            ema.update(policy)
            if (step + 1) % 250 == 0:
                print(f"{kind} step={step+1} loss={loss.item():.6f}", flush=True)
        values = {"final_train_loss": loss.item(), "seconds": time.monotonic() - started}
        with torch.no_grad():
            torch.manual_seed(1001)
            noise = torch.randn(64, 7, 8, device=device)
            cond = condition.expand(64, -1)
            for steps in ((10,) if kind == "diffusion" else (2, 4)):
                pred = ema.model.sample(noise.shape, cond, sampler="ddim" if kind == "diffusion" else "euler",
                    inference_steps=steps, initial_noise=noise)
                metrics = {"mse": (pred - target).square().mean().item(), "action_absmax": pred.abs().max().item(),
                    "gripper_mse": (pred[:, -1] - target[:, -1]).square().mean().item()}
                if kind == "flow_matching":
                    raw = ema.model.sample(noise.shape, cond, inference_steps=steps, initial_noise=noise, clamp_final=False)
                    metrics["raw_mse"] = (raw - target).square().mean().item()
                    metrics["raw_absmax"] = raw.abs().max().item()
                values[str(steps)] = metrics
        result["results"][kind] = values
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    diffusion = result["results"]["diffusion"]["10"]
    flow = result["results"]["flow_matching"]["4"]
    result["gate_pass"] = diffusion["mse"] < .005 and flow["mse"] < .005 and flow["raw_mse"] < .005
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
