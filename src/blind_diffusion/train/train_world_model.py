import os
from typing import Dict
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from omegaconf import OmegaConf
import hydra
from blind_diffusion.utils.seed import set_seed
from blind_diffusion.utils.device import get_device
from blind_diffusion.utils.logging import JSONLLogger
from blind_diffusion.utils.checkpoint import save_checkpoint
from blind_diffusion.data.robomimic_dataset import RoboMimicSequenceDataset
from blind_diffusion.models.encoders import MLPEncoder
from blind_diffusion.models.rssm import RSSM
from blind_diffusion.models.heads import RewardHead, DoneHead, ObsHead
from blind_diffusion.models.losses import kl_normal, mse_loss, bce_logits_loss


class WorldModel(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, cfg_model):
        super().__init__()
        enc_hidden = cfg_model.encoder.hidden_dim
        enc_layers = cfg_model.encoder.layers
        self.encoder = MLPEncoder(obs_dim, enc_hidden, enc_layers)
        self.rssm = RSSM(
            action_dim=action_dim,
            obs_dim=enc_hidden,
            deter_dim=cfg_model.rssm.deter_dim,
            stoch_dim=cfg_model.rssm.stoch_dim,
            hidden_dim=cfg_model.rssm.hidden_dim,
            min_std=cfg_model.rssm.min_std,
            max_std=cfg_model.rssm.max_std,
        )
        feat_dim = cfg_model.rssm.deter_dim + cfg_model.rssm.stoch_dim
        self.reward_head = RewardHead(feat_dim, cfg_model.heads.reward_hidden)
        self.done_head = DoneHead(feat_dim, cfg_model.heads.done_hidden)
        self.obs_head = ObsHead(feat_dim, obs_dim, cfg_model.heads.obs_hidden)

    def forward(self, obs_seq, action_seq):
        obs_embed = self.encoder(obs_seq)
        post = self.rssm.observe(obs_embed, action_seq)
        feat = torch.cat([post["h"], post["z"]], dim=-1)
        reward_pred = self.reward_head(feat)
        done_logits = self.done_head(feat)
        obs_pred = self.obs_head(feat)
        return post, obs_pred, reward_pred, done_logits


def compute_loss(batch, model: WorldModel, burn_in: int, kl_free_bits: float, kl_scale: float):
    obs = batch["obs"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    dones = batch["dones"]

    post, obs_pred, reward_pred, done_logits = model(obs, actions)

    start = burn_in
    obs_loss = mse_loss(obs_pred[:, start:], obs[:, start:])
    reward_loss = mse_loss(reward_pred[:, start:], rewards[:, start:])
    done_loss = bce_logits_loss(done_logits[:, start:], dones[:, start:])

    kl = kl_normal(post["post_mean"], post["post_std"], post["prior_mean"], post["prior_std"]).sum(-1)
    if kl_free_bits > 0:
        kl = torch.clamp(kl, min=kl_free_bits)
    kl_loss = kl[:, start:].mean() * kl_scale

    total = obs_loss + reward_loss + done_loss + kl_loss
    return total, {
        "loss": total.item(),
        "obs_loss": obs_loss.item(),
        "reward_loss": reward_loss.item(),
        "done_loss": done_loss.item(),
        "kl_loss": kl_loss.item(),
    }


def build_loaders(cfg):
    hdf5_path = os.path.join(os.environ.get("ROBO_DATA", ""), cfg.task.hdf5_name)
    dataset = RoboMimicSequenceDataset(
        hdf5_path=hdf5_path,
        obs_keys=cfg.task.obs_keys,
        seq_len=cfg.seq_len,
        burn_in=cfg.burn_in,
    )
    val_size = int(len(dataset) * cfg.val_fraction)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return dataset, train_loader, val_loader


@hydra.main(version_base=None, config_path="../configs", config_name="train_world_model")
def main(cfg):
    cfg_model = cfg.model

    set_seed(cfg.seed)
    device = get_device()

    dataset, train_loader, val_loader = build_loaders(cfg)
    obs_dim = dataset[0]["obs"].shape[-1]
    action_dim = dataset[0]["actions"].shape[-1]

    model = WorldModel(obs_dim, action_dim, cfg_model).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    run_dir = os.path.join(cfg.run_dir, cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    logger = JSONLLogger(os.path.join(run_dir, "logs.jsonl"))

    best_val = float("inf")
    step = 0
    pbar = tqdm(total=cfg.max_steps, desc="train_wm", leave=True)
    while step < cfg.max_steps:
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, logs = compute_loss(batch, model, cfg.burn_in, cfg.kl_free_bits, cfg.kl_scale)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            if step % cfg.log_every == 0:
                logger.log({"step": step, **logs})
            pbar.update(1)
            step += 1
            if step >= cfg.max_steps:
                break

        if step % cfg.val_every == 0:
            val_loss = evaluate(model, val_loader, device, cfg)
            logger.log({"step": step, "val_loss": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    os.path.join(run_dir, "checkpoints/best.pt"),
                    {
                        "model": model.state_dict(),
                        "config": OmegaConf.to_container(cfg, resolve=True),
                        "task": OmegaConf.to_container(cfg.task, resolve=True),
                        "norm": dataset.get_norm_stats(),
                    },
                )

    metrics = {"best_val": best_val}
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pbar.close()


def evaluate(model, loader, device, cfg):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = compute_loss(batch, model, cfg.burn_in, cfg.kl_free_bits, cfg.kl_scale)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


if __name__ == "__main__":
    main()
