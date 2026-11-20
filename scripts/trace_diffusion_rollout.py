"""Two-episode trace diagnostic for a frozen causal diffusion policy."""

import argparse
import json
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import torch

from blind_diffusion.diffusion.diffusion import GaussianDiffusion
from blind_diffusion.diffusion.model import UNet1D
from blind_diffusion.train.train_world_model import WorldModel
from blind_diffusion.utils.checkpoint import load_checkpoint
from blind_diffusion.utils.seed import set_seed
from blind_diffusion.eval.protocol import is_blackout, task_success

import robomimic.utils.obs_utils as ObsUtils


def _raw_obs(obs, keys):
    return np.concatenate([np.asarray(obs[k], dtype=np.float32) for k in keys]).tolist()


def _obs_to_vec(obs, keys):
    return torch.cat([torch.tensor(obs[k]).float() for k in keys], dim=-1)


def _normalize(x, mean, std):
    if mean is None or std is None:
        return x
    return (x - torch.tensor(mean, device=x.device, dtype=x.dtype)) / torch.tensor(std, device=x.device, dtype=x.dtype)


def select_rhc_action(sequence):
    return sequence[:, :, 0]


def make_env(hdf5_path):
    """Standalone Lift environment setup so remote snapshots need no env package."""
    if "mujoco_py" not in sys.modules:
        module = types.ModuleType("mujoco_py")
        module.builder = types.SimpleNamespace(MujocoException=Exception)
        sys.modules["mujoco_py"] = module
    with h5py.File(hdf5_path, "r") as handle:
        raw = handle["data"].attrs["env_args"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    from robomimic.utils.env_utils import create_env_from_metadata
    return create_env_from_metadata(json.loads(raw), use_image_obs=False, render_offscreen=False)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--wm-checkpoint", required=True)
    parser.add_argument("--diff-checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--blackout-period", type=int, default=20)
    parser.add_argument("--blackout-duration", type=int, default=0)
    parser.add_argument(
        "--ddim-noise-mode",
        choices=("fresh", "fixed"),
        default="fresh",
        help="Fresh terminal noise per RHC query (historical behavior), or one fixed terminal-noise chunk per episode.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    env = make_env(args.dataset)
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": obs_keys, "rgb": []}})

    initial_obs = env.reset()
    obs_dim = sum(initial_obs[key].shape[0] for key in obs_keys)
    wm_ckpt = load_checkpoint(args.wm_checkpoint, map_location=device)
    norm = wm_ckpt["norm"]
    action_dim = wm_ckpt["model"]["rssm.gru.weight_ih"].shape[1] - 30
    if env.action_dimension != action_dim:
        raise RuntimeError(f"Environment action dimension {env.action_dimension} != checkpoint action dimension {action_dim}")

    # Config dimensions are encoded in this recovery checkpoint's architecture.
    from omegaconf import OmegaConf
    model_cfg = OmegaConf.load("src/blind_diffusion/configs/model/rssm_small.yaml")
    wm = WorldModel(obs_dim, action_dim, model_cfg).to(device)
    wm.load_state_dict(wm_ckpt["model"])
    wm.eval()

    diff_ckpt = load_checkpoint(args.diff_checkpoint, map_location=device)
    diff_cfg = diff_ckpt["config"]["diffusion"]
    if diff_cfg["prediction_type"] != "x0":
        raise RuntimeError("Trace requires the bounded direct-x0 checkpoint")
    cond_dim = model_cfg.rssm.deter_dim + model_cfg.rssm.stoch_dim
    denoiser = UNet1D(action_dim, diff_cfg["horizon"], cond_dim, base_ch=diff_cfg["base_ch"],
                      architecture=diff_cfg.get("architecture", "legacy")).to(device)
    diffusion = GaussianDiffusion(denoiser, diff_cfg["timesteps"], diff_cfg["schedule"], prediction_type="x0").to(device)
    diffusion.load_state_dict(diff_ckpt.get("ema_model", diff_ckpt["model"]))
    diffusion.eval()

    episodes = []
    for episode_index in range(args.episodes):
        obs = env.reset()
        h = torch.zeros(1, model_cfg.rssm.deter_dim, device=device)
        z = torch.zeros(1, model_cfg.rssm.stoch_dim, device=device)
        prev_action = torch.zeros(1, action_dim, device=device)
        fixed_noise = (
            torch.randn(1, action_dim, diff_cfg["horizon"], device=device)
            if args.ddim_noise_mode == "fixed"
            else None
        )
        trajectory, ep_return, success, collision = [], 0.0, False, False

        for step in range(args.max_steps):
            obs_vec = _obs_to_vec(obs, obs_keys).unsqueeze(0).to(device)
            embedded = wm.encoder(_normalize(obs_vec, norm.get("obs_mean"), norm.get("obs_std")))
            blind = is_blackout(step, args.warmup_steps, args.blackout_period, args.blackout_duration)
            state = (wm.rssm.imagine_step(prev_action, {"h": h, "z": z}, deterministic=True)
                     if blind else wm.rssm.observe_step(embedded, prev_action, {"h": h, "z": z}, deterministic=True))
            h, z = state["h"], state["z"]
            belief = torch.cat([h, z], dim=-1)
            sequence = diffusion.sample(
                (1, action_dim, diff_cfg["horizon"]),
                belief,
                sampler="ddim",
                inference_steps=10,
                initial_noise=fixed_noise,
            )
            action = select_rhc_action(sequence)
            action = action.clamp(-1.0, 1.0)
            action_np = action.squeeze(0).detach().cpu().numpy()

            eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
            quat = np.asarray(obs["robot0_eef_quat"], dtype=float)
            gripper_state = np.asarray(obs["robot0_gripper_qpos"], dtype=float)
            object_obs = np.asarray(obs["object"], dtype=float)
            object_pos = object_obs[:3]
            trajectory.append({
                "step": step,
                "blackout": blind,
                "observation": _raw_obs(obs, obs_keys),
                "action": action_np.tolist(),
                "eef_pos": eef.tolist(),
                "eef_quat": quat.tolist(),
                "gripper_command": float(action_np[-1]),
                "gripper_qpos": gripper_state.tolist(),
                "object_pos": object_pos.tolist(),
                "gripper_object_distance": float(np.linalg.norm(eef - object_pos)),
                "h_norm": float(h.norm().item()),
                "z_norm": float(z.norm().item()),
            })
            obs, reward, done, info = env.step(action_np)
            trajectory[-1].update({
                "next_observation": _raw_obs(obs, obs_keys),
                "reward": float(reward),
                "done": bool(done),
                "success_after_step": task_success(env, info),
                "collision_after_step": bool(info.get("collision", False) or info.get("violation", False)),
                "info": {k: v for k, v in info.items() if isinstance(v, (bool, int, float, str))},
            })
            prev_action = action
            ep_return += float(reward)
            success = success or task_success(env, info)
            collision = collision or bool(info.get("collision", False) or info.get("violation", False))
            if done:
                break

        actions = np.asarray([x["action"] for x in trajectory])
        eef_positions = np.asarray([x["eef_pos"] for x in trajectory])
        object_positions = np.asarray([x["object_pos"] for x in trajectory])
        distances = np.asarray([x["gripper_object_distance"] for x in trajectory])
        episodes.append({
            "episode": episode_index,
            "return": ep_return,
            "success": success,
            "collision": collision,
            "length": len(trajectory),
            "summary": {
                "action_mean": actions.mean(axis=0).tolist(),
                "action_std": actions.std(axis=0).tolist(),
                "action_absmax": float(np.abs(actions).max()),
                "gripper_sign_changes": int(np.count_nonzero(np.diff(np.signbit(actions[:, -1])))),
                "eef_displacement": float(np.linalg.norm(eef_positions[-1] - eef_positions[0])),
                "object_displacement": float(np.linalg.norm(object_positions[-1] - object_positions[0])),
                "distance_initial": float(distances[0]),
                "distance_min": float(distances.min()),
                "distance_final": float(distances[-1]),
                "h_norm_start_end": [trajectory[0]["h_norm"], trajectory[-1]["h_norm"]],
                "z_norm_start_end": [trajectory[0]["z_norm"], trajectory[-1]["z_norm"]],
            },
            "trajectory": trajectory,
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"ddim_noise_mode": args.ddim_noise_mode, "episodes": episodes}, indent=2) + "\n")
    print(json.dumps({"episodes": [{k: v for k, v in episode.items() if k != "trajectory"} for episode in episodes]}, indent=2))


if __name__ == "__main__":
    main()
