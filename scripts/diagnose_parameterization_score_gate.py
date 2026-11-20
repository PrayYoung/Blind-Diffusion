#!/usr/bin/env python
"""Epsilon, x0, and v-prediction comparison on one fixed synthetic target."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from blind_diffusion.diffusion.diffusion import GaussianDiffusion
from blind_diffusion.diffusion.model import UNet1D


def targets(kind, x0, x_t, eps, alpha_bar):
    a, s = torch.sqrt(alpha_bar), torch.sqrt(1.0 - alpha_bar)
    if kind == "epsilon": return eps
    if kind == "x0": return x0
    if kind == "v": return a * eps - s * x0
    raise ValueError(kind)


def to_x0(kind, prediction, x_t, alpha_bar):
    a, s = torch.sqrt(alpha_bar), torch.sqrt(1.0 - alpha_bar)
    if kind == "epsilon": return (x_t - s * prediction) / a
    if kind == "x0": return prediction
    if kind == "v": return a * x_t - s * prediction
    raise ValueError(kind)


def run_case(kind, x0, cond, steps, batch_size, lr, eval_size):
    torch.manual_seed(13579)
    model = UNet1D(7, 8, 230, 64)
    diffusion = GaussianDiffusion(model, timesteps=100, schedule="cosine")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        t = torch.randint(0, 100, (batch_size,))
        eps = torch.randn(batch_size, 7, 8)
        x0_batch = x0.expand_as(eps)
        x_t = diffusion.q_sample(x0_batch, t, eps)
        alpha_bar = diffusion.alphas_cumprod[t].view(-1, 1, 1)
        loss = F.mse_loss(model(x_t, t, cond.expand(batch_size, -1)), targets(kind, x0_batch, x_t, eps, alpha_bar))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    g = torch.Generator().manual_seed(98765)
    t = torch.randint(0, 100, (eval_size,), generator=g)
    eps = torch.randn(eval_size, 7, 8, generator=g)
    with torch.no_grad():
        x0_batch = x0.expand_as(eps); x_t = diffusion.q_sample(x0_batch, t, eps)
        alpha_bar = diffusion.alphas_cumprod[t].view(-1, 1, 1)
        prediction = model(x_t, t, cond.expand(eval_size, -1))
        target = targets(kind, x0_batch, x_t, eps, alpha_bar)
        x0_hat = to_x0(kind, prediction, x_t, alpha_bar)
        target_mse = F.mse_loss(prediction, target, reduction="none").mean((1, 2))
    bins = {}
    for low in range(0, 100, 25):
        mask = (t >= low) & (t < low + 25)
        bins[f"{low:02d}-{low+24:02d}"] = float(target_mse[mask].mean())
    x0_mse = float(F.mse_loss(x0_hat, x0_batch))
    return {"parameterization": kind, "fresh_target_mse": float(target_mse.mean()), "fresh_target_mse_by_timestep_bin": bins, "fresh_x0_mse": x0_mse, "strong_gate_pass": x0_mse <= 1e-2}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True); p.add_argument("--steps", type=int, default=5000); p.add_argument("--batch-size", type=int, default=32); p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--eval-size", type=int, default=4000); args = p.parse_args()
    torch.manual_seed(20260820); x0 = torch.rand(1, 7, 8) * 2 - 1; cond = torch.randn(1, 230)
    result = {"protocol": {"schedule": "cosine", "timesteps": 100, "fixed_x0_shape": [1, 7, 8], "fixed_cond_shape": [1, 230], "updates": args.steps, "batch_size": args.batch_size, "optimizer": "Adam", "lr": args.lr, "strong_x0_mse_gate": 1e-2}, "cases": [run_case(kind, x0, cond, args.steps, args.batch_size, args.lr, args.eval_size) for kind in ("epsilon", "x0", "v")]}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
