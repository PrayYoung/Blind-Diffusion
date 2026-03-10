import os
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
from blind_diffusion.data.robomimic_dataset_image import RoboMimicImageSequenceDataset
from blind_diffusion.models.encoders import MLPEncoder
from blind_diffusion.models.vision_encoder import SmallResNet
from blind_diffusion.models.rssm import RSSM
from blind_diffusion.models.heads import RewardHead, DoneHead, ObsHead
from blind_diffusion.models.losses import kl_normal, mse_loss, bce_logits_loss
from blind_diffusion.models.modules import mlp


class WorldModelImage(torch.nn.Module):
    def __init__(self, image_ch, lowdim_dim, action_dim, cfg_model):
        super().__init__()
        self.image_encoder = SmallResNet(image_ch, cfg_model.vision.embed_dim)
        self.lowdim_encoder = None
        if lowdim_dim and lowdim_dim > 0:
            self.lowdim_encoder = MLPEncoder(lowdim_dim, cfg_model.vision.lowdim_embed, 2)
        enc_out = cfg_model.vision.embed_dim + (cfg_model.vision.lowdim_embed if self.lowdim_encoder else 0)

        self.rssm = RSSM(
            action_dim=action_dim,
            obs_dim=enc_out,
            deter_dim=cfg_model.rssm.deter_dim,
            stoch_dim=cfg_model.rssm.stoch_dim,
            hidden_dim=cfg_model.rssm.hidden_dim,
            min_std=cfg_model.rssm.min_std,
            max_std=cfg_model.rssm.max_std,
        )
        feat_dim = cfg_model.rssm.deter_dim + cfg_model.rssm.stoch_dim
        self.reward_head = RewardHead(feat_dim, cfg_model.heads.reward_hidden)
        self.done_head = DoneHead(feat_dim, cfg_model.heads.done_hidden)
        # predict lowdim state if provided
        self.obs_head = ObsHead(feat_dim, lowdim_dim, cfg_model.heads.obs_hidden) if lowdim_dim > 0 else None
        # predict image embedding (stop-grad target)
        self.vision_head = mlp(feat_dim, cfg_model.vision.embed_dim, cfg_model.vision_head.hidden_dim, 2)
        self.vision_loss_scale = float(cfg_model.vision_head.loss_scale)

    def encode_obs(self, images, lowdim=None):
        B, T = images.shape[:2]
        x = images.view(B * T, *images.shape[2:])
        img_embed = self.image_encoder(x).view(B, T, -1)
        if self.lowdim_encoder is None:
            return img_embed
        low = self.lowdim_encoder(lowdim.view(B * T, -1)).view(B, T, -1)
        return torch.cat([img_embed, low], dim=-1)

    def forward(self, images, actions, lowdim=None):
        obs_embed = self.encode_obs(images, lowdim)
        post = self.rssm.observe(obs_embed, actions)
        feat = torch.cat([post["h"], post["z"]], dim=-1)
        reward_pred = self.reward_head(feat)
        done_logits = self.done_head(feat)
        obs_pred = self.obs_head(feat) if self.obs_head is not None else None
        vision_pred = self.vision_head(feat)
        return post, obs_pred, reward_pred, done_logits, obs_embed, vision_pred


def compute_loss(batch, model: WorldModelImage, burn_in: int, kl_free_bits: float, kl_scale: float):
    images = batch["images"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    dones = batch["dones"]
    lowdim = batch.get("lowdim")

    post, obs_pred, reward_pred, done_logits, obs_embed, vision_pred = model(images, actions, lowdim)

    start = burn_in
    reward_loss = mse_loss(reward_pred[:, start:], rewards[:, start:])
    done_loss = bce_logits_loss(done_logits[:, start:], dones[:, start:])

    obs_loss = 0.0
    if obs_pred is not None and lowdim is not None:
        obs_loss = mse_loss(obs_pred[:, start:], lowdim[:, start:])

    kl = kl_normal(post["post_mean"], post["post_std"], post["prior_mean"], post["prior_std"]).sum(-1)
    if kl_free_bits > 0:
        kl = torch.clamp(kl, min=kl_free_bits)
    kl_loss = kl[:, start:].mean() * kl_scale

    # stop-grad target for image embedding
    vision_target = obs_embed[:, start:,:vision_pred.shape[-1]].detach()
    vision_loss = mse_loss(vision_pred[:, start:], vision_target)
    total = reward_loss + done_loss + kl_loss + obs_loss + vision_loss * model.vision_loss_scale
    return total, {
        "loss": total.item(),
        "reward_loss": reward_loss.item(),
        "done_loss": done_loss.item(),
        "kl_loss": kl_loss.item(),
        "obs_loss": float(obs_loss) if isinstance(obs_loss, float) else obs_loss.item(),
        "vision_loss": vision_loss.item(),
    }


def build_loaders(cfg):
    hdf5_path = os.path.join(os.environ.get("ROBO_DATA", ""), cfg.task.hdf5_name)
    dataset = RoboMimicImageSequenceDataset(
        hdf5_path=hdf5_path,
        image_keys=cfg.task.image_keys,
        lowdim_keys=cfg.task.get("lowdim_keys", []),
        seq_len=cfg.seq_len,
        burn_in=cfg.burn_in,
        augment=cfg.get("augment", False),
        crop_size=cfg.get("crop_size", None),
        image_size=cfg.task.get("image_size", None),
    )
    val_size = int(len(dataset) * cfg.val_fraction)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return dataset, train_loader, val_loader


@hydra.main(version_base=None, config_path="../configs", config_name="train_world_model_image")
def main(cfg):
    model_cfg = cfg.model

    set_seed(cfg.seed)
    device = get_device()

    dataset, train_loader, val_loader = build_loaders(cfg)
    image_ch = dataset[0]["images"].shape[1]
    lowdim_dim = dataset[0].get("lowdim").shape[-1] if "lowdim" in dataset[0] else 0
    action_dim = dataset[0]["actions"].shape[-1]

    model = WorldModelImage(image_ch, lowdim_dim, action_dim, model_cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    run_dir = os.path.join(cfg.run_dir, cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    logger = JSONLLogger(os.path.join(run_dir, "logs.jsonl"))

    best_val = float("inf")
    step = 0
    pbar = tqdm(total=cfg.max_steps, desc="train_wm_image", leave=True)
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
