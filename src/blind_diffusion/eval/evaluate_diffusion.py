import os
import json
import numpy as np
import torch
from tqdm import tqdm
import hydra
from blind_diffusion.utils.seed import set_seed
from blind_diffusion.utils.device import get_device
from blind_diffusion.utils.metrics import summarize_episode_metrics
from blind_diffusion.utils.checkpoint import load_checkpoint
from blind_diffusion.env.robomimic_env import make_env
from blind_diffusion.train.train_world_model import WorldModel
from blind_diffusion.diffusion.model import UNet1D
from blind_diffusion.diffusion.diffusion import GaussianDiffusion

import robomimic.utils.obs_utils as ObsUtils


def _obs_to_vec(obs, obs_keys):
    return torch.cat([torch.tensor(obs[k]).float() for k in obs_keys], dim=-1)

def _normalize(x: torch.Tensor, mean, std):
    if mean is None or std is None:
        return x
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype)
    std_t = torch.tensor(std, device=x.device, dtype=x.dtype)
    return (x - mean_t) / std_t


def _denormalize(x: torch.Tensor, mean, std):
    if mean is None or std is None:
        return x
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype)
    std_t = torch.tensor(std, device=x.device, dtype=x.dtype)
    return x * std_t + mean_t


@hydra.main(version_base=None, config_path="../configs", config_name="eval_diffusion")
def main(cfg):

    set_seed(cfg.seed)
    device = get_device()

    hdf5_path = os.path.join(os.environ.get("ROBO_DATA", ""), cfg.task.hdf5_name)
    env = make_env(hdf5_path)

    rgb_keys = [k for k in cfg.task.obs_keys if "rgb" in k.lower() or "image" in k.lower()]
    low_dim_keys = [k for k in cfg.task.obs_keys if k not in rgb_keys]

    ObsUtils.initialize_obs_utils_with_obs_specs({
        "obs": {
            "low_dim": low_dim_keys,
            "rgb": rgb_keys
        }
    })

    # world model
    dummy_obs = env.reset()
    obs_dim = sum([dummy_obs[k].shape[0] for k in cfg.task.obs_keys])
    act_dim = env.action_dimension
    wm = WorldModel(obs_dim, act_dim, cfg.model).to(device)
    wm_ckpt = load_checkpoint(cfg.wm_checkpoint, map_location=device)
    wm.load_state_dict(wm_ckpt["model"])
    wm.eval()
    norm = wm_ckpt.get("norm", {})
    obs_mean = norm.get("obs_mean")
    obs_std = norm.get("obs_std")
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

    max_steps = getattr(env, "_max_episode_steps", None) or getattr(env, "horizon", None) or cfg.max_steps
    episodes = []
    mode = cfg.get("eval", {}).get("mode", "rhc")
    for _ in tqdm(range(cfg.episodes), desc="eval_diffusion", leave=True):
        obs = env.reset()
        done = False
        success = 0.0
        collision = 0.0
        ep_return = 0.0
        steps = 0

        h = torch.zeros(1, cfg.model.rssm.deter_dim, device=device)
        z = torch.zeros(1, cfg.model.rssm.stoch_dim, device=device)
        prev_action = torch.zeros(1, act_dim, device=device)

        open_loop_plan = None
        open_loop_idx = 0
        while not done and steps < max_steps:
            obs_vec = _obs_to_vec(obs, cfg.task.obs_keys).unsqueeze(0).to(device)
            obs_vec_n = _normalize(obs_vec, obs_mean, obs_std)
            obs_embed = wm.encoder(obs_vec_n)

            state = wm.rssm.observe_step(obs_embed, prev_action, {"h": h, "z": z})
            h, z = state["h"], state["z"]
            belief = torch.cat([h, z], dim=-1)
            if mode == "open_loop":
                if open_loop_plan is None or open_loop_idx >= cfg.diffusion.horizon:
                    open_loop_plan = diff.sample((1, act_dim, cfg.diffusion.horizon), belief)
                    open_loop_idx = 0
                action = open_loop_plan[:, :, open_loop_idx]
                open_loop_idx += 1
            else:
                seq = diff.sample((1, act_dim, cfg.diffusion.horizon), belief)
                action = seq[:, :, 0]

            action_env = _denormalize(action, act_mean, act_std)
            action_np = action_env.squeeze(0).detach().cpu().numpy()
            obs, reward, done, info = env.step(action_np)
            prev_action = action

            ep_return += float(reward)
            steps += 1
            if info.get("success", False):
                success = 1.0
            if info.get("collision", False) or info.get("violation", False):
                collision = 1.0

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
    with open(os.path.join(run_dir, "diffusion_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
