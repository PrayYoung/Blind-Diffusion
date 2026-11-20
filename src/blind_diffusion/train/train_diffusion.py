import os
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
from blind_diffusion.data.robomimic_dataset import build_episode_datasets
from blind_diffusion.train.train_world_model import WorldModel
from blind_diffusion.train.beliefs import compute_pre_action_beliefs, sample_conditioning_queries
from blind_diffusion.utils.ema import ModelEMA
from blind_diffusion.diffusion.policy import build_action_policy


def policy_batch_loss(diffusion, wm, batch, cfg, augment=True):
    obs, actions = batch["obs"], batch["actions"]
    valid, is_start = batch["valid"], batch["is_start"]
    starts, blackout = sample_conditioning_queries(valid, is_start, cfg.diffusion.horizon, cfg.burn_in,
        cfg.prior_aug_prob if augment else 0.0, cfg.warmup_steps, cfg.blackout_max_duration)
    with torch.no_grad():
        beliefs = compute_pre_action_beliefs(wm, obs, actions, deterministic=True, blackout=blackout)
    indices = starts[:, None] + torch.arange(cfg.diffusion.horizon, device=obs.device)
    target = actions.gather(1, indices[:, :, None].expand(-1, -1, actions.shape[-1])).transpose(1, 2)
    mask = valid.gather(1, indices).unsqueeze(1)
    cond = beliefs[torch.arange(obs.shape[0], device=obs.device), starts]
    return diffusion.loss(target, cond, mask=mask)


@torch.no_grad()
def evaluate_policy(diffusion, wm, loader, device, cfg):
    total, count = 0.0, 0
    # Repeat the same Monte Carlo draws for comparable checkpoint selection.
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg.seed + 10000)
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            for _ in range(cfg.val_mc_samples):
                loss = policy_batch_loss(diffusion, wm, batch, cfg, augment=False)
                total += loss.item() * len(batch["obs"])
                count += len(batch["obs"])
    return total / count


@hydra.main(version_base=None, config_path="../configs", config_name="train_diffusion")
def main(cfg):

    set_seed(cfg.seed)
    device = get_device()

    hdf5_path = os.path.join(os.environ.get("ROBO_DATA", ""), cfg.task.hdf5_name)
    ckpt = load_checkpoint(cfg.wm_checkpoint, map_location=device)
    if "split" not in ckpt:
        raise ValueError("Retrain the WM with an episode split before corrected policy training")
    dataset, val_dataset, split = build_episode_datasets(
        hdf5_path=hdf5_path,
        obs_keys=cfg.task.obs_keys,
        seq_len=cfg.seq_len,
        burn_in=cfg.burn_in,
        normalize_obs=cfg.normalize_obs,
        normalize_action=cfg.normalize_action,
        pad_end=cfg.pad_end,
        val_fraction=cfg.val_fraction, seed=cfg.seed,
        norm_stats=ckpt["norm"], split=ckpt["split"],
    )
    pin = (get_device().type == "cuda")
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=pin
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=pin)

    obs_dim = dataset[0]["obs"].shape[-1]
    act_dim = dataset[0]["actions"].shape[-1]

    # load frozen world model
    wm = WorldModel(obs_dim, act_dim, cfg.model).to(device)
    wm.load_state_dict(ckpt["model"])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    cond_dim = cfg.model.rssm.deter_dim + cfg.model.rssm.stoch_dim
    diffusion = build_action_policy(cfg, act_dim, cond_dim).to(device)

    optim = torch.optim.AdamW(diffusion.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema = ModelEMA(diffusion, cfg.ema_decay)
    total_steps = cfg.epochs * len(loader)
    if cfg.max_train_steps is not None:
        total_steps = min(total_steps, cfg.max_train_steps)
    if total_steps < 1:
        raise ValueError("Policy training requires at least one optimization step")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, total_steps)

    run_dir = os.path.join(cfg.run_dir, cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    logger = JSONLLogger(os.path.join(run_dir, "diffusion_logs.jsonl"))

    step = 0
    best_epoch_loss = float("inf")
    best_epoch = None
    best_val = float("inf")
    pbar = tqdm(total=total_steps, desc="train_diffusion", leave=False, dynamic_ncols=True)
    for epoch in range(cfg.epochs):
        epoch_losses = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = policy_batch_loss(diffusion, wm, batch, cfg)
            epoch_losses.append(loss.item())

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), cfg.grad_clip)
            optim.step()
            scheduler.step()
            ema.update(diffusion)

            if step % cfg.log_every == 0:
                logger.log({"step": step, "loss": loss.item()})
            step += 1
            pbar.update(1)
            if cfg.max_train_steps is not None and step >= cfg.max_train_steps:
                break

        save_checkpoint(
            os.path.join(run_dir, "checkpoints/last.pt"),
            {
                "model": diffusion.state_dict(),
                "ema_model": ema.model.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "task": OmegaConf.to_container(cfg.task, resolve=True),
            },
        )
        mean_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        val_loss = evaluate_policy(ema.model, wm, val_loader, device, cfg)
        logger.log({"epoch": epoch, "epoch_loss": mean_epoch_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_epoch_loss = mean_epoch_loss
            best_epoch = epoch
            save_checkpoint(
                os.path.join(run_dir, "checkpoints/best.pt"),
                {
                    "model": diffusion.state_dict(),
                    "ema_model": ema.model.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "task": OmegaConf.to_container(cfg.task, resolve=True),
                    "wm_checkpoint": cfg.wm_checkpoint,
                    "split": split,
                    "selection": {"metric": "ema_heldout_velocity_loss" if cfg.get("policy_type") == "flow_matching" else "ema_heldout_denoising_loss", "value": best_val, "epoch": best_epoch},
                },
            )
        if cfg.max_train_steps is not None and step >= cfg.max_train_steps:
            break

    # A bounded smoke may stop before an epoch boundary; it still needs an
    # explicit artifact to validate the subsequent evaluator stage.
    if cfg.max_train_steps is not None and step >= cfg.max_train_steps:
        save_checkpoint(
            os.path.join(run_dir, "checkpoints/diffusion_smoke.pt"),
            {
                "model": diffusion.state_dict(),
                "ema_model": ema.model.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "task": OmegaConf.to_container(cfg.task, resolve=True),
                "wm_checkpoint": cfg.wm_checkpoint,
            },
        )

    with open(os.path.join(run_dir, "diffusion_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"final_loss": float(loss.item()), "best_val_loss": best_val, "best_epoch_loss": best_epoch_loss, "best_epoch": best_epoch},
            f,
            indent=2,
        )
    pbar.close()


if __name__ == "__main__":
    main()
