import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from blind_diffusion.data.robomimic_dataset_image import RoboMimicImageSequenceDataset
from blind_diffusion.models.vision_encoder import SmallResNet
from blind_diffusion.models.modules import mlp
from blind_diffusion.utils.device import get_device
from blind_diffusion.utils.logging import JSONLLogger
from blind_diffusion.utils.checkpoint import save_checkpoint


class BCImagePolicy(torch.nn.Module):
    def __init__(self, image_ch: int, lowdim_dim: int, action_dim: int, embed_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.image_encoder = SmallResNet(image_ch, embed_dim)
        self.lowdim_dim = lowdim_dim
        in_dim = embed_dim + (lowdim_dim if lowdim_dim > 0 else 0)
        self.head = mlp(in_dim, action_dim, hidden_dim, 3)

    def forward(self, images: torch.Tensor, lowdim: torch.Tensor | None = None) -> torch.Tensor:
        # images: [B, C, H, W]
        img_embed = self.image_encoder(images)
        if lowdim is None:
            feat = img_embed
        else:
            feat = torch.cat([img_embed, lowdim], dim=-1)
        return self.head(feat)


def train_bc_image(cfg, task_cfg, run_dir):
    hdf5_path = os.path.join(os.environ.get("ROBO_DATA", ""), task_cfg.hdf5_name)
    dataset = RoboMimicImageSequenceDataset(
        hdf5_path=hdf5_path,
        image_keys=task_cfg.image_keys,
        lowdim_keys=task_cfg.get("lowdim_keys", []),
        seq_len=cfg.seq_len,
        burn_in=0,
        normalize_action=True,
        normalize_lowdim=True,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    image_ch = dataset[0]["images"].shape[1]
    lowdim_dim = dataset[0].get("lowdim").shape[-1] if "lowdim" in dataset[0] else 0
    act_dim = dataset[0]["actions"].shape[-1]

    device = get_device()
    policy = BCImagePolicy(image_ch, lowdim_dim, act_dim, embed_dim=cfg.embed_dim, hidden_dim=cfg.hidden_dim).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    logger = JSONLLogger(os.path.join(run_dir, "bc_image_logs.jsonl"))
    step = 0
    total_steps = cfg.bc_epochs * len(loader)
    pbar = tqdm(total=total_steps, desc="train_bc_image", leave=True)
    for epoch in range(cfg.bc_epochs):
        for batch in loader:
            images = batch["images"][:, 0].to(device)
            actions = batch["actions"][:, 0].to(device)
            lowdim = batch.get("lowdim")
            if lowdim is not None:
                lowdim = lowdim[:, 0].to(device)
            pred = policy(images, lowdim)
            loss = torch.mean((pred - actions) ** 2)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step % cfg.log_every == 0:
                logger.log({"step": step, "loss": loss.item()})
            step += 1
            pbar.update(1)

    save_checkpoint(
        os.path.join(run_dir, "checkpoints/bc_image.pt"),
        {"model": policy.state_dict(), "norm": dataset.get_norm_stats()},
    )
    with open(os.path.join(run_dir, "bc_image_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"loss": float(loss.item())}, f, indent=2)
    pbar.close()
    return policy
