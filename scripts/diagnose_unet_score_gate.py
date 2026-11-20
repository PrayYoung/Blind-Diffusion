#!/usr/bin/env python
"""Offline synthetic score-field gate; it never loads a dataset or environment.

The target is fixed.  Each optimizer update uses a fresh random DDPM timestep
and noise, so success requires learning epsilon(x_t, t) away from memorized
tuples.  The Antigravity UNet is deliberately reproduced below rather than
imported, to make an implementation comparison executable.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from blind_diffusion.diffusion.diffusion import GaussianDiffusion
from blind_diffusion.diffusion.model import UNet1D as ProductionUNet1D


class AntigravitySinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1) if self.dim % 2 else emb


class AntigravityFiLM(nn.Module):
    def __init__(self, cond_dim, channels):
        super().__init__(); self.net = nn.Sequential(nn.Linear(cond_dim, channels * 2), nn.SiLU(), nn.Linear(channels * 2, channels * 2))
    def forward(self, x, cond):
        scale, shift = torch.chunk(self.net(cond), 2, dim=-1)
        return x * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)


class AntigravityResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1); self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.film = AntigravityFiLM(cond_dim, out_ch); self.act = nn.SiLU()
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, cond):
        h = self.act(self.conv1(x)); h = self.film(h, cond); h = self.act(self.conv2(h))
        return h + self.skip(x)


class AntigravityUNet1D(nn.Module):
    def __init__(self, act_dim, horizon, cond_dim, base_ch=64):
        super().__init__(); self.horizon = horizon
        self.time_emb = AntigravitySinusoidalTimeEmbedding(base_ch)
        self.time_mlp = nn.Sequential(nn.Linear(base_ch, base_ch * 2), nn.SiLU(), nn.Linear(base_ch * 2, base_ch))
        self.down1 = AntigravityResBlock1D(act_dim, base_ch, cond_dim + base_ch)
        self.down2 = AntigravityResBlock1D(base_ch, base_ch * 2, cond_dim + base_ch)
        self.pool = nn.AvgPool1d(2); self.mid = AntigravityResBlock1D(base_ch * 2, base_ch * 2, cond_dim + base_ch)
        self.up = nn.Upsample(scale_factor=2, mode="nearest"); self.up1 = AntigravityResBlock1D(base_ch * 2, base_ch, cond_dim + base_ch)
        self.out = nn.Conv1d(base_ch, act_dim, 1)
    def forward(self, x, t, cond):
        t_emb = self.time_mlp(self.time_emb(t)); cond_all = torch.cat([cond, t_emb], dim=-1)
        d1 = self.down1(x, cond_all); d2 = self.down2(self.pool(d1), cond_all)
        return self.out(self.up1(self.up(self.mid(d2, cond_all)), cond_all))


class TimestepAffineScore(nn.Module):
    """Exact function family for one fixed x0: eps = a_t*x_t + b_t."""
    def __init__(self, act_dim, horizon, cond_dim, timesteps=100):
        super().__init__(); del cond_dim
        self.scale = nn.Embedding(timesteps, act_dim * horizon)
        self.bias = nn.Embedding(timesteps, act_dim * horizon)
        nn.init.zeros_(self.scale.weight); nn.init.zeros_(self.bias.weight)
        self.act_dim, self.horizon = act_dim, horizon
    def forward(self, x, t, cond):
        del cond
        shape = (x.shape[0], self.act_dim, self.horizon)
        return self.scale(t).view(shape) * x + self.bias(t).view(shape)


def finite_range(tensor):
    return {"min": float(tensor.min()), "max": float(tensor.max()), "absmax": float(tensor.abs().max())}


def run(name, model, x0, cond, steps, batch_size, lr, eval_t, eval_noise):
    diffusion = GaussianDiffusion(model, timesteps=100, schedule="cosine")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    activations = {"time_mlp": [], "film": []}
    hooks = []
    for module_name, module in model.named_modules():
        if module_name == "time_mlp":
            hooks.append(module.register_forward_hook(lambda _m, _i, out: activations["time_mlp"].append(finite_range(out.detach()))))
        if module_name.endswith("film"):
            hooks.append(module.register_forward_hook(lambda _m, _i, out: activations["film"].append(finite_range(out.detach()))))
    grad_norms, losses = [], []
    for _ in range(steps):
        t = torch.randint(0, 100, (batch_size,))
        noise = torch.randn(batch_size, *x0.shape[1:])
        xt = diffusion.q_sample(x0.expand_as(noise), t, noise)
        pred = model(xt, t, cond.expand(batch_size, -1))
        loss = F.mse_loss(pred, noise)
        optimizer.zero_grad(); loss.backward()
        total = sum((p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None), torch.zeros(()))
        grad_norms.append(float(total.sqrt())); optimizer.step(); losses.append(float(loss.detach()))
    for hook in hooks: hook.remove()
    with torch.no_grad():
        xt = diffusion.q_sample(x0.expand_as(eval_noise), eval_t, eval_noise)
        pred = model(xt, eval_t, cond.expand(len(eval_t), -1))
        fresh = F.mse_loss(pred, eval_noise, reduction="none").mean((1, 2))
        x0_hat = diffusion.predict_x0_from_eps(xt, eval_t, pred)
        by_bin = {}
        for lo in range(0, 100, 25):
            mask = (eval_t >= lo) & (eval_t < lo + 25)
            by_bin[f"{lo:02d}-{lo + 24:02d}"] = float(fresh[mask].mean())
    def aggregate(key):
        values = activations[key]
        return None if not values else {"min": min(v["min"] for v in values), "max": max(v["max"] for v in values), "absmax": max(v["absmax"] for v in values)}
    return {"name": name, "parameter_count": sum(p.numel() for p in model.parameters()), "train_epsilon_mse_last_100": sum(losses[-100:]) / min(100, len(losses)), "fresh_epsilon_mse": float(fresh.mean()), "fresh_epsilon_mse_by_timestep_bin": by_bin, "fresh_x0_reconstruction_mse": float(F.mse_loss(x0_hat, x0.expand_as(x0_hat))), "gradient_norm": {"first": grad_norms[0], "last": grad_norms[-1], "min": min(grad_norms), "max": max(grad_norms)}, "activation_ranges": {"time_mlp": aggregate("time_mlp"), "film_output": aggregate("film")}}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--steps", type=int, default=5000); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--output", required=True); args = parser.parse_args()
    torch.set_num_threads(8); torch.manual_seed(20260820)
    x0 = torch.rand(1, 7, 8) * 2 - 1; cond = torch.randn(1, 230)
    generator = torch.Generator().manual_seed(98765)
    eval_t = torch.randint(0, 100, (4000,), generator=generator); eval_noise = torch.randn(4000, 7, 8, generator=generator)
    specs = [("production_unet", lambda: ProductionUNet1D(7, 8, 230, 64), args.lr), ("antigravity_unet", lambda: AntigravityUNet1D(7, 8, 230, 64), args.lr), ("timestep_affine_baseline", lambda: TimestepAffineScore(7, 8, 230), 1e-2)]
    result = {"protocol": {"fixed_x0_range": finite_range(x0), "fixed_condition_shape": list(cond.shape), "timesteps": 100, "schedule": "cosine", "objective": "epsilon MSE", "updates_per_model": args.steps, "batch_size": args.batch_size, "optimizer": "Adam", "lr": args.lr, "fresh_eval_examples": len(eval_t), "bins": ["00-24", "25-49", "50-74", "75-99"]}, "models": []}
    for name, constructor, model_lr in specs:
        # Equal initialization and identical fresh-noise/timestep draws for A/B.
        torch.manual_seed(13579)
        result["models"].append(run(name, constructor(), x0, cond, args.steps, args.batch_size, model_lr, eval_t, eval_noise))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
