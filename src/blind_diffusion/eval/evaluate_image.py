import os
import json
import warnings
import numpy as np
import torch
from tqdm import tqdm
import hydra
from blind_diffusion.utils.seed import set_seed
from blind_diffusion.utils.device import get_device
from blind_diffusion.utils.metrics import summarize_episode_metrics
from blind_diffusion.utils.checkpoint import load_checkpoint
from blind_diffusion.env.robomimic_env import make_env
from blind_diffusion.utils.video import save_video
from blind_diffusion.train.train_world_model_image import WorldModelImage
from blind_diffusion.diffusion.model import UNet1D
from blind_diffusion.diffusion.diffusion import GaussianDiffusion
from blind_diffusion.baselines.open_loop_bc_image import BCImagePolicy
import robomimic.utils.obs_utils as ObsUtils


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


def _sensor_block_active(step: int, cfg) -> bool:
    if not cfg.get("sensor_block", {}).get("enable", False):
        return False
    every_k = cfg.sensor_block.get("every_k", 20)
    duration = cfg.sensor_block.get("duration", 5)
    if every_k <= 0 or duration <= 0:
        return False
    return (step % every_k) < duration


def _apply_sensor_block(img: torch.Tensor, step: int, cfg) -> torch.Tensor:
    if not _sensor_block_active(step, cfg):
        return img
    mode = cfg.sensor_block.get("mode", "zero")
    if mode == "noise":
        return torch.rand_like(img)
    return torch.zeros_like(img)


@hydra.main(version_base=None, config_path="../configs", config_name="eval_image")
def main(cfg):

    set_seed(cfg.seed)
    device = get_device()

    hdf5_path = os.path.join(os.environ.get("ROBO_DATA", ""), cfg.task.hdf5_name)
    ObsUtils.initialize_obs_utils_with_obs_specs({
        "obs": {
            "low_dim":cfg.task.get("lowdim_keys", []),
            "rgb": cfg.task.get("image_keys", []),
        }
    })
    env = make_env(hdf5_path, image_keys=cfg.task.image_keys, image_size=cfg.task.get("image_size", 84))
    dummy_obs = env.reset()
    # world model
    act_dim = env.action_dimension
    image_ch = len(cfg.task.image_keys) * 3
    lowdim_keys = cfg.task.get("lowdim_keys", [])
    lowdim_dim = sum([dummy_obs[k].shape[0] for k in lowdim_keys]) if lowdim_keys else 0

    wm = WorldModelImage(image_ch, lowdim_dim, act_dim, cfg.model).to(device)
    wm_ckpt = load_checkpoint(cfg.wm_checkpoint, map_location=device)
    wm.load_state_dict(wm_ckpt["model"])
    wm.eval()
    norm = wm_ckpt.get("norm", {})
    low_mean = norm.get("low_mean")
    low_std = norm.get("low_std")
    act_mean = norm.get("act_mean")
    act_std = norm.get("act_std")

    # diffusion
    cond_dim = cfg.model.rssm.deter_dim + cfg.model.rssm.stoch_dim
    unet = UNet1D(act_dim, cfg.diffusion.horizon, cond_dim, base_ch=cfg.diffusion.base_ch).to(device)
    diff = GaussianDiffusion(
        unet, timesteps=cfg.diffusion.timesteps, schedule=cfg.diffusion.schedule
    ).to(device)
    diff_ckpt = load_checkpoint(cfg.diff_checkpoint, map_location=device)
    diff.load_state_dict(diff_ckpt["model"])
    diff.eval()

    run_dir = os.path.join(cfg.run_dir, cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)

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
    if mode.startswith("image_"):
        mode = mode.replace("image_", "", 1)

    bc_policy = None
    if mode == "bc_eval":
        bc_ckpt = load_checkpoint(os.path.join(cfg.run_dir, cfg.exp_name, "checkpoints/bc_image.pt"), map_location=device)
        bc_policy = BCImagePolicy(image_ch, lowdim_dim, act_dim).to(device)
        bc_policy.load_state_dict(bc_ckpt["model"])
        bc_policy.eval()
        bc_norm = bc_ckpt.get("norm", {})
        act_mean = bc_norm.get("act_mean", act_mean)
        act_std = bc_norm.get("act_std", act_std)
        low_mean = bc_norm.get("low_mean", low_mean)
        low_std = bc_norm.get("low_std", low_std)

    warned_proxy = False
    max_steps = getattr(env, "_max_episode_steps", None) or getattr(env, "horizon", None) or cfg.max_steps
    for ep in tqdm(range(cfg.episodes), desc="eval_image", leave=True):
        obs = env.reset()
        done = False
        success = 0.0
        collision = 0.0
        ep_return = 0.0
        steps = 0
        frames = []

        h = torch.zeros(1, cfg.model.rssm.deter_dim, device=device)
        z = torch.zeros(1, cfg.model.rssm.stoch_dim, device=device)
        prev_action = torch.zeros(1, act_dim, device=device)

        open_loop_plan = None
        open_loop_idx = 0

        while not done and steps < max_steps:
            imgs = [obs[k] for k in cfg.task.image_keys]
            img = np.concatenate(imgs, axis=-1)
            if cfg.eval.get("save_video", False):
                frames.append(img.copy())
            img = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            block_active = _sensor_block_active(steps, cfg.eval)
            if block_active and cfg.eval.sensor_block.get("mode", "zero") == "prior":
                x = torch.cat([z, prev_action], dim=-1)
                h = wm.rssm.gru(x, h)
                prior_params = wm.rssm.prior_net(h)
                prior_mean, prior_std = wm.rssm._get_stats(prior_params)
                z = wm.rssm._sample(prior_mean, prior_std)
            else:
                img = _apply_sensor_block(img, steps, cfg.eval)
                low = _obs_to_lowdim(obs, cfg.task.get("lowdim_keys", []))
                if low is not None:
                    low = normalize(low.to(device), low_mean, low_std).unsqueeze(0)
                obs_embed = wm.encode_obs(img.unsqueeze(0), low.unsqueeze(0) if low is not None else None)
                state = wm.rssm.observe_step(obs_embed.squeeze(1), prev_action, {"h": h, "z": z})
                h, z = state["h"], state["z"]
            belief = torch.cat([h, z], dim=-1)

            if mode == "bc_eval":
                action = bc_policy(img, low if low is not None else None)
                action = action.unsqueeze(0)
            elif mode == "open_loop":
                if open_loop_plan is None or open_loop_idx >= cfg.horizon:
                    open_loop_plan = diff.sample((1, act_dim, cfg.diffusion.horizon), belief)
                    open_loop_idx = 0
                action = open_loop_plan[:, :, open_loop_idx]
                open_loop_idx += 1
            else:
                seq = diff.sample((1, act_dim, cfg.diffusion.horizon), belief)
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

        if cfg.eval.get("save_video", False):
            video_path = os.path.join(run_dir, f"video_ep{ep}.mp4")
            save_video(frames, video_path, fps=cfg.eval.get("video_fps", 30))

    metrics = summarize_episode_metrics(episodes)
    metrics["avg_return"] = float(np.mean([e["return"] for e in episodes])) if episodes else 0.0
    metrics["avg_length"] = float(np.mean([e["length"] for e in episodes])) if episodes else 0.0

    with open(os.path.join(run_dir, "image_diffusion_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
