import os
import json
import numpy as np
import torch
from omegaconf import OmegaConf

from blind_diffusion.utils.hydra import parse_config
from blind_diffusion.utils.seed import set_seed
from blind_diffusion.utils.device import get_device
from blind_diffusion.utils.metrics import summarize_episode_metrics
from blind_diffusion.utils.checkpoint import load_checkpoint
from blind_diffusion.env.robomimic_env import make_env
from blind_diffusion.train.train_world_model import WorldModel
from blind_diffusion.diffusion.model import UNet1D
from blind_diffusion.diffusion.diffusion import GaussianDiffusion


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
    obs_dim = sum([env.observation_space[k].shape[0] for k in task_cfg.obs_keys])
    act_dim = env.action_space.shape[0]
    wm = WorldModel(obs_dim, act_dim, model_cfg).to(device)
    wm_ckpt = load_checkpoint(cfg.wm_checkpoint, map_location=device)
    wm.load_state_dict(wm_ckpt["model"])
    wm.eval()
    norm = wm_ckpt.get("norm", {})
    obs_mean = norm.get("obs_mean")
    obs_std = norm.get("obs_std")
    act_mean = norm.get("act_mean")
    act_std = norm.get("act_std")

    # diffusion
    cond_dim = model_cfg.rssm.deter_dim + model_cfg.rssm.stoch_dim
    unet = UNet1D(act_dim, cfg.horizon, cond_dim, base_ch=cfg.base_ch).to(device)
    diff = GaussianDiffusion(unet, timesteps=cfg.timesteps, schedule=cfg.schedule).to(device)
    diff_ckpt = load_checkpoint(cfg.diff_checkpoint, map_location=device)
    diff.load_state_dict(diff_ckpt["model"])
    diff.eval()

    episodes = []
    mode = cfg.get("eval", {}).get("mode", "rhc")
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
            obs_vec = _obs_to_vec(obs, task_cfg.obs_keys).unsqueeze(0).to(device)
            obs_vec_n = _normalize(obs_vec, obs_mean, obs_std)
            obs_embed = wm.encoder(obs_vec_n)

            state = wm.rssm.observe_step(obs_embed, prev_action, {"h": h, "z": z})
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
