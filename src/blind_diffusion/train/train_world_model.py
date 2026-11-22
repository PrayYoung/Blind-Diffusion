import os
from typing import Dict
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf
import hydra
from blind_diffusion.utils.seed import set_seed
from blind_diffusion.utils.device import get_device
from blind_diffusion.utils.logging import JSONLLogger
from blind_diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from blind_diffusion.utils.training_state import signature, make_state, restore_state, reconcile_log
from blind_diffusion.data.robomimic_dataset import build_episode_datasets
from blind_diffusion.models.encoders import MLPEncoder
from blind_diffusion.models.rssm import RSSM
from blind_diffusion.models.heads import RewardHead, DoneHead, ObsHead
from blind_diffusion.models.losses import kl_normal, mse_loss, bce_logits_loss
from blind_diffusion.train.beliefs import previous_actions


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
        # state[t] is the posterior belief for obs[t], after the transition
        # caused by action[t - 1]. This is also the policy decision state for
        # choosing action[t].
        obs_embed = self.encoder(obs_seq)
        post = self.rssm.observe(obs_embed, previous_actions(action_seq))
        feat = torch.cat([post["h"], post["z"]], dim=-1)
        reward_pred = self.reward_head(feat)
        done_logits = self.done_head(feat)
        obs_pred = self.obs_head(feat)
        return post, obs_pred, reward_pred, done_logits


def compute_loss(batch, model: WorldModel, burn_in: int, kl_free_bits: float, kl_scale: float,
                 obs_noise_std: float = 0.0, previous_action_noise_std: float = 0.0,
                 obs_weight=1.0, imag_horizon=0, imag_weight=0.0, overshoot_weight=0.0,
                 deterministic=False):
    obs = batch["obs"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    dones = batch["dones"]

    # The legacy objective is entirely expert-forced.  When enabled, corrupt
    # only the *belief inputs* but retain clean targets: this teaches recovery
    # rather than asking the decoder to model artificial noise.
    obs_input = obs + torch.randn_like(obs) * obs_noise_std if obs_noise_std else obs
    action_inputs = previous_actions(actions)
    if previous_action_noise_std:
        action_inputs = action_inputs + torch.randn_like(action_inputs) * previous_action_noise_std
        action_inputs[:, 0] = 0.0
    obs_embed = model.encoder(obs_input)
    post = model.rssm.observe(obs_embed, action_inputs, deterministic=deterministic)
    feat = torch.cat([post["h"], post["z"]], dim=-1)
    obs_pred = model.obs_head(feat)
    reward_pred = model.reward_head(feat)
    done_logits = model.done_head(feat)

    # state[t] reconstructs obs[t]; state[t + 1] is the first state after
    # action[t], so reward/done[t] are decoded from that next state.
    state_start = burn_in + 1
    valid = batch.get("valid", torch.ones(obs.shape[:2], device=obs.device, dtype=torch.bool))
    def masked(error, mask):
        while mask.ndim < error.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(error).to(error.dtype)
        return (error * mask).sum() / mask.sum().clamp_min(1)
    obs_loss = masked((obs_pred[:, state_start:] - obs[:, state_start:]).square(), valid[:, state_start:])
    transition_valid = valid[:, state_start:] & valid[:, burn_in:-1]
    reward_loss = masked((reward_pred[:, state_start:] - rewards[:, burn_in:-1]).square(), transition_valid)
    done_loss = masked(torch.nn.functional.binary_cross_entropy_with_logits(
        done_logits[:, state_start:], dones[:, burn_in:-1], reduction="none"), valid[:, state_start:])

    kl = kl_normal(post["post_mean"], post["post_std"], post["prior_mean"], post["prior_std"]).sum(-1)
    if kl_free_bits > 0:
        kl = torch.clamp(kl, min=kl_free_bits)
    kl_loss = masked(kl[:, state_start:], valid[:, state_start:]) * kl_scale

    # Start from every eligible posterior, then recursively use only priors.
    # Predicted state at t+k consumes action[t+k-1], never action[t+k].
    imag_loss = obs_loss.new_zeros(())
    overshoot_loss = obs_loss.new_zeros(())
    horizon = min(imag_horizon, obs.shape[1] - burn_in - 1)
    if horizon > 0:
        starts = obs.shape[1] - horizon - burn_in
        state = {key: post[key][:, burn_in:burn_in + starts].detach().reshape(-1, post[key].shape[-1]) for key in ("h", "z")}
        alive = valid[:, burn_in:burn_in + starts].clone()
        for offset in range(1, horizon + 1):
            action = actions[:, burn_in + offset - 1:burn_in + offset - 1 + starts]
            alive = alive & valid[:, burn_in + offset:burn_in + offset + starts] & ~dones[:, burn_in + offset - 1:burn_in + offset - 1 + starts].bool()
            state = model.rssm.imagine_step(action.reshape(-1, actions.shape[-1]), state, deterministic=True)
            feature = torch.cat([state["h"], state["z"]], -1)
            target_obs = obs[:, burn_in + offset:burn_in + offset + starts]
            imag_loss = imag_loss + masked((model.obs_head(feature).reshape_as(target_obs) - target_obs).square(), alive)
            target_h = post["h"][:, burn_in + offset:burn_in + offset + starts].detach()
            target_mean = post["post_mean"][:, burn_in + offset:burn_in + offset + starts].detach()
            target_std = post["post_std"][:, burn_in + offset:burn_in + offset + starts].detach()
            overshoot_loss = overshoot_loss + masked((state["h"].reshape_as(target_h) - target_h).square(), alive)
            overshoot_loss = overshoot_loss + 0.05 * masked(kl_normal(target_mean, target_std,
                state["prior_mean"].reshape_as(target_mean), state["prior_std"].reshape_as(target_std)).sum(-1), alive)
        imag_loss, overshoot_loss = imag_loss / horizon, overshoot_loss / horizon

    total = obs_weight * obs_loss + reward_loss + done_loss + kl_loss + imag_weight * imag_loss + overshoot_weight * overshoot_loss
    return total, {
        "loss": total.item(),
        "obs_loss": obs_loss.item(),
        "reward_loss": reward_loss.item(),
        "done_loss": done_loss.item(),
        "kl_loss": kl_loss.item(),
        "imag_obs_loss": imag_loss.item(),
        "overshoot_loss": overshoot_loss.item(),
    }


def build_loaders(cfg):
    hdf5_path = os.path.join(os.environ.get("ROBO_DATA", ""), cfg.task.hdf5_name)
    dataset, val_ds, split = build_episode_datasets(
        hdf5_path=hdf5_path,
        obs_keys=cfg.task.obs_keys,
        seq_len=cfg.seq_len,
        burn_in=cfg.burn_in,
        normalize_obs=cfg.normalize_obs,
        normalize_action=cfg.normalize_action,
        pad_end=cfg.pad_end,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )
    # Splitting random windows leaks adjacent states from one demonstration
    # into both partitions. Hold out complete demonstrations instead.
    dataset.split = split

    pin = (get_device().type == "cuda")
    train_loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin
    )
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
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, cfg.max_steps)

    run_dir = os.path.join(cfg.run_dir, cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    logger = JSONLLogger(os.path.join(run_dir, "logs.jsonl"))

    best_val = float("inf")
    step,epochs = 0,0
    best_path=os.path.join(run_dir,"checkpoints/best.pt")
    hdf5_path=os.path.join(os.environ.get("ROBO_DATA",""),cfg.task.hdf5_name)
    training_signature=signature(OmegaConf.to_container(cfg,resolve=True),hdf5_path)
    if cfg.get("resume_checkpoint"):
        saved=load_checkpoint(cfg.resume_checkpoint,map_location=device)
        model.load_state_dict(saved["model"])
        state=restore_state(saved,optim,scheduler,training_signature,best_path)
        step,epochs,best_val=state["step"],state["epoch"],state["best_val"]
        reconcile_log(os.path.join(run_dir,"logs.jsonl"),step,epochs)
    pbar = tqdm(total=cfg.max_steps,initial=step,desc="train_wm",leave=False,dynamic_ncols=True)
    while step < cfg.max_steps:
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, logs = compute_loss(
                batch, model, cfg.burn_in, cfg.kl_free_bits, cfg.kl_scale,
                cfg.robustness.obs_noise_std, cfg.robustness.previous_action_noise_std,
                cfg.obs_weight, cfg.imag_horizon, cfg.imag_weight, cfg.overshoot_weight,
            )

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            scheduler.step()

            if step % cfg.log_every == 0:
                logger.log({"step": step, **logs})
            pbar.update(1)
            step += 1
            if step % cfg.val_every == 0 or step == cfg.max_steps:
                validation = evaluate(model, val_loader, device, cfg)
                val_loss = validation["loss"]
                logger.log({"step": step, **{f"val_{key}": value for key, value in validation.items()}})
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(os.path.join(run_dir, "checkpoints/best.pt"), {
                        "model": model.state_dict(), "config": OmegaConf.to_container(cfg, resolve=True),
                        "task": OmegaConf.to_container(cfg.task, resolve=True),
                        "norm": dataset.get_norm_stats(), "split": dataset.split,
                    })
                model.train()
            if step >= cfg.max_steps:
                break
        epochs += 1
        save_checkpoint(os.path.join(run_dir,"checkpoints/last.pt"),{
            "model":model.state_dict(),"config":OmegaConf.to_container(cfg,resolve=True),
            "norm":dataset.get_norm_stats(),"split":dataset.split,
            "training_state":make_state(optim,scheduler,training_signature,best_path,
                                         step=step,epoch=epochs,best_val=best_val)})

    metrics = {"best_val": best_val}
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pbar.close()


def evaluate(model, loader, device, cfg):
    model.eval()
    totals, count = {}, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, metrics = compute_loss(batch, model, cfg.burn_in, cfg.kl_free_bits, cfg.kl_scale,
                obs_weight=cfg.obs_weight, imag_horizon=cfg.imag_horizon,
                imag_weight=cfg.imag_weight, overshoot_weight=cfg.overshoot_weight, deterministic=True)
            size = len(batch["obs"])
            count += size
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * size
    if not count:
        raise ValueError("Validation loader is empty")
    return {key: value / count for key, value in totals.items()}


if __name__ == "__main__":
    main()
