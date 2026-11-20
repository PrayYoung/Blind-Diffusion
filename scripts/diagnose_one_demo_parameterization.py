#!/usr/bin/env python
"""One-real-demo action-chunk overfit and DDIM reconstruction gate.

This is intentionally conditioned on one fixed vector: it isolates the
diffusion target/sampler path from world-model representation learning.
"""
import argparse
import json
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F

from blind_diffusion.diffusion.diffusion import GaussianDiffusion
from blind_diffusion.diffusion.model import UNet1D


def sample_x0(model, diffusion, cond, shape):
    x = torch.randn(shape)
    schedule = torch.linspace(99, 0, 100).round().long()
    for index, t_scalar in enumerate(schedule):
        t = torch.full((shape[0],), int(t_scalar), dtype=torch.long)
        alpha_t = diffusion.alphas_cumprod[t_scalar]
        alpha_prev = diffusion.alphas_cumprod[schedule[index + 1]] if index + 1 < len(schedule) else torch.ones(())
        x0 = model(x, t, cond)
        eps = (x - torch.sqrt(alpha_t) * x0) / torch.sqrt(1.0 - alpha_t)
        x = torch.sqrt(alpha_prev) * x0 + torch.sqrt((1.0 - alpha_prev).clamp_min(0.0)) * eps
    return x


def main():
    p = argparse.ArgumentParser(); p.add_argument("--dataset", required=True); p.add_argument("--output", required=True); p.add_argument("--steps", type=int, default=5000); args = p.parse_args()
    torch.manual_seed(20260820)
    with h5py.File(args.dataset, "r") as h5:
        name = sorted(h5["data"].keys())[0]
        actions = torch.tensor(h5["data"][name]["actions"][:8], dtype=torch.float32)
    x0 = actions.transpose(0, 1).unsqueeze(0); cond = torch.randn(1, 230)
    model = UNet1D(7, 8, 230, 64); diffusion = GaussianDiffusion(model, 100, "cosine"); opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(args.steps):
        t = torch.randint(0, 100, (32,)); x0_batch = x0.expand(32, -1, -1); noise = torch.randn_like(x0_batch)
        x_t = diffusion.q_sample(x0_batch, t, noise)
        loss = F.mse_loss(model(x_t, t, cond.expand(32, -1)), x0_batch)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        fresh_t = torch.randint(0, 100, (512,)); fresh_x = torch.randn(512, 7, 8)
        fresh_mse = float(F.mse_loss(model(fresh_x, fresh_t, cond.expand(512, -1)), x0.expand(512, -1, -1)))
        torch.manual_seed(123); sample = sample_x0(model, diffusion, cond, x0.shape)
    result = {"demo": name, "target_action_range": [float(x0.min()), float(x0.max())], "fresh_x0_mse": fresh_mse, "ddim100_sample_mse": float(F.mse_loss(sample, x0)), "sample_action_range": [float(sample.min()), float(sample.max())], "sample_absmax": float(sample.abs().max()), "gate_pass": fresh_mse <= 1e-3 and float(F.mse_loss(sample, x0)) <= 1e-3 and float(sample.abs().max()) <= 1.001}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
