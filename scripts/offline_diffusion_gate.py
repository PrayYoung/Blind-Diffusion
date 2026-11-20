#!/usr/bin/env python
"""Offline-only causal WM/diffusion sanity gate; never creates an environment."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from blind_diffusion.diffusion.diffusion import GaussianDiffusion
from blind_diffusion.diffusion.model import UNet1D
from blind_diffusion.models.encoders import MLPEncoder
from blind_diffusion.models.heads import DoneHead, ObsHead, RewardHead
from blind_diffusion.models.losses import kl_normal
from blind_diffusion.models.rssm import RSSM
from blind_diffusion.train.beliefs import compute_pre_action_beliefs


class SmallWorldModel(torch.nn.Module):
    def __init__(self, obs_dim=19, action_dim=7):
        super().__init__()
        self.encoder = MLPEncoder(obs_dim, 64, 2)
        self.rssm = RSSM(action_dim, 64, 96, 16, 96)
        self.obs_head = ObsHead(112, obs_dim, 64)
        self.reward_head = RewardHead(112, 64)
        self.done_head = DoneHead(112, 64)

    def forward(self, obs, actions):
        previous = torch.cat([torch.zeros_like(actions[:, :1]), actions[:, :-1]], dim=1)
        post = self.rssm.observe(self.encoder(obs), previous)
        feat = torch.cat([post["h"], post["z"]], dim=-1)
        return post, self.obs_head(feat), self.reward_head(feat), self.done_head(feat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wm-steps", type=int, default=500)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(8)

    keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    with h5py.File(args.dataset, "r") as h5:
        names = sorted(h5["data"].keys())
        train_names, held_names = names[:160], names[160:]
        def windows(demos):
            rows = []
            for name in demos:
                group = h5["data"][name]
                for start in range(group["actions"].shape[0] - 49):
                    rows.append((np.concatenate([group["obs"][key][start:start + 50] for key in keys], axis=-1), group["actions"][start:start + 50], group["rewards"][start:start + 50], group["dones"][start:start + 50]))
            return rows
        train, held = windows(train_names), windows(held_names)
    train_obs = np.stack([row[0] for row in train]).astype(np.float32)
    mean, std = train_obs.reshape(-1, 19).mean(0), train_obs.reshape(-1, 19).std(0) + 1e-6
    def tensors(rows):
        return tuple(torch.tensor(np.stack([row[i] for row in rows]), dtype=torch.float32) for i in range(4))
    train_o, train_a, train_r, train_d = tensors(train)
    held_o, held_a, _, _ = tensors(held)
    train_o, held_o = (train_o - torch.tensor(mean)) / torch.tensor(std), (held_o - torch.tensor(mean)) / torch.tensor(std)

    wm = SmallWorldModel()
    opt = torch.optim.Adam(wm.parameters(), lr=3e-4)
    for _ in range(args.wm_steps):
        ids = torch.randint(len(train), (32,))
        post, obs_p, reward_p, done_p = wm(train_o[ids], train_a[ids])
        # Same causal state convention as production: state[t+1] follows action[t].
        kl = kl_normal(post["post_mean"], post["post_std"], post["prior_mean"], post["prior_std"]).sum(-1).clamp_min(1.0)
        loss = F.mse_loss(obs_p[:, 6:], train_o[ids, 6:]) + F.mse_loss(reward_p[:, 6:], train_r[ids, 5:-1]) + F.binary_cross_entropy_with_logits(done_p[:, 6:], train_d[ids, 5:-1]) + kl[:, 6:].mean()
        opt.zero_grad(); loss.backward(); opt.step()

    for param in wm.parameters(): param.requires_grad = False
    wm.eval()
    diff = GaussianDiffusion(UNet1D(7, 8, 112, 32), timesteps=100, schedule="cosine")
    opt = torch.optim.Adam(diff.parameters(), lr=1e-4)
    for _ in range(args.diffusion_steps):
        ids = torch.randint(len(train), (32,)); t0 = torch.randint(0, 43, (32,))
        with torch.no_grad(): beliefs = compute_pre_action_beliefs(wm, train_o[ids], train_a[ids])
        target = torch.stack([train_a[ids[i], t0[i]:t0[i] + 8] for i in range(32)]).transpose(1, 2)
        cond = torch.stack([beliefs[i, t0[i]] for i in range(32)])
        loss = diff.loss(target, cond)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        count = min(128, len(held)); t0 = torch.arange(count) % 43
        beliefs = compute_pre_action_beliefs(wm, held_o[:count], held_a[:count])
        target = torch.stack([held_a[i, t0[i]:t0[i] + 8] for i in range(count)])
        cond = torch.stack([beliefs[i, t0[i]] for i in range(count)])
        torch.manual_seed(123); pred = diff.sample((count, 7, 8), cond, sampler="ddim", inference_steps=10).transpose(1, 2)
    p, y = pred.numpy(), target.numpy()
    corr = [float(np.corrcoef(p[..., i].ravel(), y[..., i].ravel())[0, 1]) for i in range(6)]
    mse = float(np.mean((p-y)**2))
    mean_correlation = float(np.mean(corr))
    metrics = {"status": "PASS" if np.abs(p).max() <= 1.00001 and mse < 0.552154839 and mean_correlation >= 0.10 else "FAIL", "heldout_demos": len(held_names), "heldout_sequences": count, "wm_steps": args.wm_steps, "diffusion_steps": args.diffusion_steps, "ddim10_mse": mse, "ddim10_mean_correlation": mean_correlation, "generated_absmax": float(np.abs(p).max()), "gate": {"max_abs_action": 1.00001, "max_mse": 0.552154839, "min_mean_correlation": 0.10}, "broken_checkpoint_ddim10_mse": 0.552154839}
    Path(args.output).write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
