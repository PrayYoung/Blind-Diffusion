import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf
import hydra
from blind_diffusion.utils.seed import set_seed
from blind_diffusion.utils.device import get_device
from blind_diffusion.utils.logging import JSONLLogger
from blind_diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from blind_diffusion.data.robomimic_dataset_image import RoboMimicImageSequenceDataset
from blind_diffusion.train.train_world_model_image import WorldModelImage
from blind_diffusion.diffusion.model import UNet1D
from blind_diffusion.diffusion.diffusion import GaussianDiffusion


def compute_beliefs(wm: WorldModelImage, images: torch.Tensor, actions: torch.Tensor, lowdim=None):
    obs_embed = wm.encode_obs(images, lowdim)
    post = wm.rssm.observe(obs_embed, actions)
    belief = torch.cat([post["h"], post["z"]], dim=-1)
    return belief


@hydra.main(version_base=None, config_path="../../../configs", config_name="train_diffusion_image")
def main(cfg):

    set_seed(cfg.seed)
    device = get_device()

    hdf5_path = os.path.join(os.environ.get("ROBO_DATA", ""), cfg.task.hdf5_name)
    dataset = RoboMimicImageSequenceDataset(
        hdf5_path=hdf5_path,
        image_keys=cfg.task.image_keys,
        lowdim_keys=cfg.task.get("lowdim_keys", []),
        seq_len=cfg.seq_len,
        burn_in=cfg.burn_in,
        image_size=cfg.task.get("image_size", None),
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    lowdim_dim = dataset[0].get("lowdim").shape[-1] if "lowdim" in dataset[0] else 0
    act_dim = dataset[0]["actions"].shape[-1]

    # load frozen world model
    wm = WorldModelImage(dataset[0]["images"].shape[1], lowdim_dim, act_dim, cfg.model).to(device)
    ckpt = load_checkpoint(cfg.wm_checkpoint, map_location=device)
    wm.load_state_dict(ckpt["model"])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    cond_dim = cfg.model.rssm.deter_dim + cfg.model.rssm.stoch_dim
    unet = UNet1D(act_dim, cfg.diffusion.horizon, cond_dim, base_ch=cfg.diffusion.base_ch).to(device)
    diffusion = GaussianDiffusion(
        unet, timesteps=cfg.diffusion.timesteps, schedule=cfg.diffusion.schedule
    ).to(device)

    optim = torch.optim.Adam(diffusion.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    run_dir = os.path.join(cfg.run_dir, cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    logger = JSONLLogger(os.path.join(run_dir, "diffusion_logs.jsonl"))

    step = 0
    total_steps = cfg.epochs * len(loader)
    pbar = tqdm(total=total_steps, desc="train_diffusion_image", leave=True)
    for epoch in range(cfg.epochs):
        for batch in loader:
            images = batch["images"].to(device)
            actions = batch["actions"].to(device)
            lowdim = batch.get("lowdim")
            if lowdim is not None:
                lowdim = lowdim.to(device)

            with torch.no_grad():
                belief = compute_beliefs(wm, images, actions, lowdim)

            B, T, _ = actions.shape
            t0 = torch.randint(0, T - cfg.diffusion.horizon + 1, (B,), device=device)
            target = torch.stack(
                [actions[b, t0[b]: t0[b] + cfg.diffusion.horizon] for b in range(B)], dim=0
            )
            cond = torch.stack([belief[b, t0[b]] for b in range(B)], dim=0)

            target = target.transpose(1, 2)
            loss = diffusion.loss(target, cond)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), cfg.grad_clip)
            optim.step()

            if step % cfg.log_every == 0:
                logger.log({"step": step, "loss": loss.item()})
            step += 1
            pbar.update(1)

        save_checkpoint(
            os.path.join(run_dir, f"checkpoints/diffusion_epoch_{epoch}.pt"),
            {
                "model": diffusion.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "task": OmegaConf.to_container(cfg.task, resolve=True),
            },
        )

    with open(os.path.join(run_dir, "diffusion_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"final_loss": float(loss.item())}, f, indent=2)
    pbar.close()


if __name__ == "__main__":
    main()
