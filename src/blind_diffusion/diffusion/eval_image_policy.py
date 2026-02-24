import os
import json
import warnings
import numpy as np
import torch
from omegaconf import OmegaConf

from blind_diffusion.utils.hydra import parse_config
from blind_diffusion.utils.seed import set_seed
from blind_diffusion.utils.device import get_device
from blind_diffusion.utils.metrics import summarize_episode_metrics
from blind_diffusion.utils.checkpoint import load_checkpoint
from blind_diffusion.env.robomimic_env import make_env
from blind_diffusion.train.train_world_model_image import WorldModelImage
from blind_diffusion.diffusion.model import UNet1D
from blind_diffusion.diffusion.diffusion import GaussianDiffusion


def _extract_collision(info: dict) -> float:
    for k in ["collision", "contact", "n_contacts", "num_contacts", "contacts"]:
        if k in info:
            val = info[k]
            if isinstance(val, (int, float)):
                return float(val > 0)
            if isinstance(val, (list, tuple)):
                return float(len(val) > 0)
    return -1.0


def _collision_proxy_from_lowdim(lowdim: torch.Tensor, threshold: float = 5.0) -> float:
    if lowdim is None:
        return 0.0
    return float((lowdim.abs() > threshold).any().item())


def _obs_to_lowdim(obs, keys):
    if not keys:
        return None
    return torch.cat([torch.tensor(obs[k]).float() for k in keys], dim=-1)


def main():
    cfg = parse_config()
    task_cfg = OmegaConf.load(os.path.join("configs/task", f"{cfg.task}.yaml"))
    model_cfg = OmegaConf.load(os.path.join("configs/model", f"{cfg.model}.yaml"))
    diff_cfg = OmegaConf.load(os.path.join("configs/diffusion", f"{cfg.diffusion}.yaml"))
    cfg = OmegaConf.merge(cfg, diff_cfg)

    set_seed(cfg.seed)
    device = get_device()

    hdf5_path = os.path.join(os.environ.get("ROBO_DATA", ""), task_cfg.hdf5_name)
    env = make_env(hdf5_path)

    # world model
    act_dim = env.action_space.shape[0]
    image_ch = len(task_cfg.image_keys) * 3
    lowdim_dim = sum([env.observation_space[k].shape[0] for k in task_cfg.get("lowdim_keys", [])]) if task_cfg.get("lowdim_keys") else 0

    wm = WorldModelImage(image_ch, lowdim_dim, act_dim, model_cfg).to(device)
    wm_ckpt = load_checkpoint(cfg.wm_checkpoint, map_location=device)
    wm.load_state_dict(wm_ckpt["model"])
    wm.eval()
    norm = wm_ckpt.get("norm", {})
    low_mean = norm.get("low_mean")
    low_std = norm.get("low_std")
    act_mean = norm.get("act_mean")
    act_std = norm.get("act_std")

    # diffusion
    cond_dim = model_cfg.rssm.deter_dim + model_cfg.rssm.stoch_dim
    unet = UNet1D(act_dim, cfg.horizon, cond_dim, base_ch=cfg.base_ch).to(device)
    diff = GaussianDiffusion(unet, timesteps=cfg.timesteps, schedule=cfg.schedule).to(device)
    diff_ckpt = load_checkpoint(cfg.diff_checkpoint, map_location=device)
    diff.load_state_dict(diff_ckpt["model"])
    diff.eval()

    def normalize(x, mean, std):
        if mean is None or std is None:
            return x
        mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype)
        std_t = torch.tensor(std, device=x.device, dtype=x.dtype)
        return (x - mean_t) / std_t

    def denormalize(x, mean, std):
        if mean is None or std is None:
            return x
        mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype)
        std_t = torch.tensor(std, device=x.device, dtype=x.dtype)
        return x * std_t + mean_t

    episodes = []
    mode = cfg.get("eval", {}).get("mode", "rhc")

    warned_proxy = False
    for _ in range(cfg.episodes):
        obs = env.reset()
        done = False
        success = 0.0
        collision = 0.0
        ep_return = 0.0
        steps = 0

        h = torch.zeros(1, model_cfg.rssm.deter_dim, device=device)
        z = torch.zeros(1, model_cfg.rssm.stoch_dim, device=device)
        prev_action = torch.zeros(1, act_dim, device=device)

        open_loop_plan = None
        open_loop_idx = 0

        while not done:
            imgs = [obs[k] for k in task_cfg.image_keys]
            img = np.concatenate(imgs, axis=-1)
            img = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

            low = _obs_to_lowdim(obs, task_cfg.get("lowdim_keys", []))
            if low is not None:
                low = normalize(low.to(device), low_mean, low_std).unsqueeze(0)

            obs_embed = wm.encode_obs(img.unsqueeze(0), low.unsqueeze(0) if low is not None else None)
            state = wm.rssm.observe_step(obs_embed.squeeze(1), prev_action, {"h": h, "z": z})
            h, z = state["h"], state["z"]
            belief = torch.cat([h, z], dim=-1)

            if mode == "open_loop":
                if open_loop_plan is None or open_loop_idx >= cfg.horizon:
                    open_loop_plan = diff.sample((1, act_dim, cfg.horizon), belief)
                    open_loop_idx = 0
                action = open_loop_plan[:, :, open_loop_idx]
                open_loop_idx += 1
            else:
                seq = diff.sample((1, act_dim, cfg.horizon), belief)
                action = seq[:, :, 0]

            action_env = denormalize(action, act_mean, act_std).squeeze(0).cpu().numpy()
            obs, reward, done, info = env.step(action_env)
            prev_action = action

            ep_return += float(reward)
            steps += 1
            if info.get("success", False):
                success = 1.0
            col = _extract_collision(info)
            if col >= 0:
                collision = max(collision, col)
            else:
                if not warned_proxy:
                    warnings.warn("No collision/contact info in env info; using low-dim proxy threshold.", RuntimeWarning)
                    warned_proxy = True
                collision = max(collision, _collision_proxy_from_lowdim(low if low is not None else None))

        episodes.append({
            "success": success,
            "collision": collision,
            "return": ep_return,
            "length": steps,
        })

    metrics = summarize_episode_metrics(episodes)
    metrics["avg_return"] = float(np.mean([e["return"] for e in episodes])) if episodes else 0.0
    metrics["avg_length"] = float(np.mean([e["length"] for e in episodes])) if episodes else 0.0

    run_dir = os.path.join(cfg.run_dir, cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "image_diffusion_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
