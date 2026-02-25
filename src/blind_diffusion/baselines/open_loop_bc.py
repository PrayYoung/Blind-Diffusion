import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

from blind_diffusion.data.robomimic_dataset import RoboMimicSequenceDataset
from blind_diffusion.models.modules import mlp
from blind_diffusion.utils.device import get_device
from blind_diffusion.utils.logging import JSONLLogger
from blind_diffusion.utils.checkpoint import save_checkpoint


class BCPolicy(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = mlp(obs_dim, action_dim, hidden_dim, 3)

    def forward(self, obs):
        return self.net(obs)


def train_bc(cfg, task_cfg, run_dir):
    hdf5_path = os.path.join(os.environ.get("ROBO_DATA", ""), task_cfg.hdf5_name)
    dataset = RoboMimicSequenceDataset(hdf5_path, task_cfg.obs_keys, seq_len=cfg.seq_len, burn_in=0)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    obs_dim = dataset[0]["obs"].shape[-1]
    act_dim = dataset[0]["actions"].shape[-1]

    device = get_device()
    policy = BCPolicy(obs_dim, act_dim).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    logger = JSONLLogger(os.path.join(run_dir, "bc_logs.jsonl"))
    step = 0
    total_steps = cfg.bc_epochs * len(loader)
    pbar = tqdm(total=total_steps, desc="train_bc", leave=True)
    for epoch in range(cfg.bc_epochs):
        for batch in loader:
            obs = batch["obs"][:, 0].to(device)
            act = batch["actions"][:, 0].to(device)
            pred = policy(obs)
            loss = torch.mean((pred - act) ** 2)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step % cfg.log_every == 0:
                logger.log({"step": step, "loss": loss.item()})
            step += 1
            pbar.update(1)

    save_checkpoint(
        os.path.join(run_dir, "checkpoints/bc.pt"),
        {"model": policy.state_dict(), "norm": dataset.get_norm_stats()},
    )
    with open(os.path.join(run_dir, "bc_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"loss": float(loss.item())}, f, indent=2)
    pbar.close()
    return policy
