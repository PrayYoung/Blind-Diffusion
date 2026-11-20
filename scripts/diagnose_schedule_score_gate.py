#!/usr/bin/env python
"""Controlled schedule sensitivity test for the synthetic one-target score gate."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from blind_diffusion.diffusion.diffusion import GaussianDiffusion
from blind_diffusion.diffusion.model import UNet1D


def run_case(name, schedule, max_t, x0, cond, steps, batch_size, lr, eval_size):
    torch.manual_seed(13579)
    model = UNet1D(7, 8, 230, 64)
    diffusion = GaussianDiffusion(model, timesteps=100, schedule=schedule)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        t = torch.randint(0, max_t + 1, (batch_size,))
        noise = torch.randn(batch_size, 7, 8)
        x_t = diffusion.q_sample(x0.expand_as(noise), t, noise)
        loss = F.mse_loss(model(x_t, t, cond.expand(batch_size, -1)), noise)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    generator = torch.Generator().manual_seed(98765)
    t = torch.randint(0, max_t + 1, (eval_size,), generator=generator)
    noise = torch.randn(eval_size, 7, 8, generator=generator)
    with torch.no_grad():
        x_t = diffusion.q_sample(x0.expand_as(noise), t, noise)
        pred = model(x_t, t, cond.expand(eval_size, -1))
        eps_mse = F.mse_loss(pred, noise, reduction="none").mean((1, 2))
        x0_hat = diffusion.predict_x0_from_eps(x_t, t, pred)
    bins = {}
    for low in range(0, max_t + 1, 25):
        mask = (t >= low) & (t <= min(low + 24, max_t))
        bins[f"{low:02d}-{min(low + 24, max_t):02d}"] = float(eps_mse[mask].mean())
    x0_mse = float(F.mse_loss(x0_hat, x0.expand_as(x0_hat)))
    return {"name": name, "schedule": schedule, "max_training_and_eval_timestep": max_t,
            "alpha_bar_at_max_t": float(diffusion.alphas_cumprod[max_t]),
            "fresh_epsilon_mse": float(eps_mse.mean()), "fresh_epsilon_mse_by_timestep_bin": bins,
            "fresh_x0_mse": x0_mse, "strong_gate_pass": x0_mse <= 1e-2 and max(bins.values()) <= 0.05}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=5000); parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--eval-size", type=int, default=4000); args = parser.parse_args()
    torch.manual_seed(20260820)
    x0 = torch.rand(1, 7, 8) * 2 - 1; cond = torch.randn(1, 230)
    cases = [("cosine_full", "cosine", 99), ("linear_full", "linear", 99), ("cosine_capped_t89", "cosine", 89)]
    result = {"protocol": {"fixed_x0_shape": [1, 7, 8], "fixed_cond_shape": [1, 230], "objective": "epsilon", "updates": args.steps, "batch_size": args.batch_size, "lr": args.lr, "strong_gate": {"max_x0_mse": 1e-2, "max_timestep_bin_epsilon_mse": 0.05}}, "cases": []}
    for case in cases: result["cases"].append(run_case(*case, x0, cond, args.steps, args.batch_size, args.lr, args.eval_size))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
