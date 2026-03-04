import os
import json
import torch
from omegaconf import OmegaConf
import numpy as np
import robomimic.utils.obs_utils as ObsUtils
import hydra
from blind_diffusion.utils.seed import set_seed
from blind_diffusion.utils.device import get_device
from blind_diffusion.utils.metrics import summarize_episode_metrics
from blind_diffusion.utils.checkpoint import load_checkpoint
from blind_diffusion.env.robomimic_env import make_env
from blind_diffusion.planner.mpc import build_planner
from blind_diffusion.train.train_world_model import WorldModel
from blind_diffusion.baselines.open_loop_bc import BCPolicy, train_bc
from blind_diffusion.baselines.open_loop_random import RandomPolicy


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


@hydra.main(version_base=None, config_path="../../configs", config_name="eval_mpc")
def main(cfg):
    set_seed(cfg.seed)

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

    mode = cfg.get("eval", {}).get("mode", "mpc")
    if mode == "bc_train":
        run_dir = os.path.join(cfg.run_dir, cfg.exp_name)
        os.makedirs(run_dir, exist_ok=True)
        train_bc(cfg, cfg.task, run_dir)
        return

    device = get_device()
    ckpt = load_checkpoint(cfg.checkpoint, map_location=device)

    dummy_obs = env.reset()
    obs_dim = sum([dummy_obs[k].shape[0] for k in low_dim_keys])
    action_dim = env.action_dimension

    wm = WorldModel(obs_dim, action_dim, cfg.model).to(device)
    wm.load_state_dict(ckpt["model"])
    wm.eval()

    planner = build_planner(cfg, action_dim)
    norm = ckpt.get("norm", {})
    obs_mean = norm.get("obs_mean")
    obs_std = norm.get("obs_std")
    act_mean = norm.get("act_mean")
    act_std = norm.get("act_std")

    action_low = torch.tensor(-np.ones(action_dim), device=device, dtype=torch.float32)
    action_high = torch.tensor(np.ones(action_dim), device=device, dtype=torch.float32)

    if mode == "bc_eval":
        bc = BCPolicy(obs_dim, action_dim).to(device)
        bc_ckpt = load_checkpoint(os.path.join(cfg.run_dir, cfg.exp_name, "checkpoints/bc.pt"), map_location=device)
        bc.load_state_dict(bc_ckpt["model"])
        bc.eval()
        bc_norm = bc_ckpt.get("norm", {})
        obs_mean = bc_norm.get("obs_mean")
        obs_std = bc_norm.get("obs_std")
        act_mean = bc_norm.get("act_mean")
        act_std = bc_norm.get("act_std")

        episodes = []
        for _ in range(cfg.episodes):
            obs = env.reset()
            done = False
            success = 0.0
            collision = 0.0
            while not done:
                obs_vec = _obs_to_vec(obs, low_dim_keys).to(device)
                obs_vec = _normalize(obs_vec, obs_mean, obs_std)
                action = bc(obs_vec).detach()
                action_env = _denormalize(action, act_mean, act_std)
                action_env = torch.clamp(action_env, action_low, action_high).cpu().numpy()
                obs, reward, done, info = env.step(action_env)
                if info.get("success", False):
                    success = 1.0
                if info.get("collision", False) or info.get("violation", False):
                    collision = 1.0
            episodes.append({"success": success, "collision": collision})
    elif mode == "random":
        policy = RandomPolicy(action_dim, low=float(action_low.min()), high=float(action_high.max()))
        episodes = []
        for _ in range(cfg.episodes):
            obs = env.reset()
            done = False
            success = 0.0
            collision = 0.0
            while not done:
                action_env = policy(None).numpy()
                obs, reward, done, info = env.step(action_env)
                if info.get("success", False):
                    success = 1.0
                if info.get("collision", False) or info.get("violation", False):
                    collision = 1.0
            episodes.append({"success": success, "collision": collision})
    else:
        episodes = []
        for _ in range(cfg.episodes):
            obs = env.reset()
            done = False
            success = 0.0
            collision = 0.0

            # initial latent via a single-step posterior
            obs_vec = _obs_to_vec(obs, low_dim_keys).unsqueeze(0).unsqueeze(0).to(device)
            obs_vec_n = _normalize(obs_vec, obs_mean, obs_std)
            act_zero = torch.zeros(1, 1, action_dim, device=device)
            post = wm.rssm.observe(wm.encoder(obs_vec_n), act_zero)
            state = {"h": post["h"][:, -1], "z": post["z"][:, -1]}

            while not done:
                action_n = planner.plan(state, wm, cfg.planner.terminal_penalty, cfg.planner.constraint_penalty)
                action_env = _denormalize(action_n, act_mean, act_std)
                action_env = torch.clamp(action_env, action_low, action_high).detach().cpu().numpy()
                obs, reward, done, info = env.step(action_env)

                obs_vec = _obs_to_vec(obs, low_dim_keys).unsqueeze(0).unsqueeze(0).to(device)
                obs_vec_n = _normalize(obs_vec, obs_mean, obs_std)
                act_t = action_n.float().unsqueeze(0).unsqueeze(0).to(device)
                post = wm.rssm.observe(wm.encoder(obs_vec_n), act_t)
                state = {"h": post["h"][:, -1], "z": post["z"][:, -1]}

                if info.get("success", False):
                    success = 1.0
                if info.get("collision", False) or info.get("violation", False):
                    collision = 1.0

            episodes.append({"success": success, "collision": collision})

    metrics = summarize_episode_metrics(episodes)
    run_dir = os.path.join(cfg.run_dir, cfg.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
