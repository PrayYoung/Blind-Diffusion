#!/usr/bin/env python3
"""Compare full-episode and training-window RSSM conditioning offline.

Diffusion training receives a fresh-zero RSSM state for each 50-step dataset
window. Evaluation retains the state through a 500-step episode. This audit
uses demonstration observations and actions for both constructions, so it
isolates the history-length mismatch from policy-induced observation shift.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from blind_diffusion.diffusion.diffusion import GaussianDiffusion
from blind_diffusion.diffusion.model import UNet1D
from blind_diffusion.models.encoders import MLPEncoder
from blind_diffusion.models.heads import DoneHead, ObsHead, RewardHead
from blind_diffusion.models.rssm import RSSM
from blind_diffusion.utils.checkpoint import load_checkpoint


OBS_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")


class WorldModel(torch.nn.Module):
    """Checkpoint-compatible model definition without Hydra training imports."""

    def __init__(self, obs_dim, action_dim, cfg_model):
        super().__init__()
        self.encoder = MLPEncoder(obs_dim, cfg_model.encoder.hidden_dim, cfg_model.encoder.layers)
        self.rssm = RSSM(
            action_dim=action_dim,
            obs_dim=cfg_model.encoder.hidden_dim,
            deter_dim=cfg_model.rssm.deter_dim,
            stoch_dim=cfg_model.rssm.stoch_dim,
            hidden_dim=cfg_model.rssm.hidden_dim,
            min_std=cfg_model.rssm.min_std,
            max_std=cfg_model.rssm.max_std,
        )
        feature_dim = cfg_model.rssm.deter_dim + cfg_model.rssm.stoch_dim
        self.reward_head = RewardHead(feature_dim, cfg_model.heads.reward_hidden)
        self.done_head = DoneHead(feature_dim, cfg_model.heads.done_hidden)
        self.obs_head = ObsHead(feature_dim, obs_dim, cfg_model.heads.obs_hidden)


def observe_mean_step(rssm, embedding, previous_action, state):
    """Causal posterior update using its mean to eliminate sampling confounds."""
    h = rssm.gru(torch.cat([state["z"], previous_action], dim=-1), state["h"])
    mean, _ = rssm._get_stats(rssm.post_net(torch.cat([h, embedding], dim=-1)))
    return {"h": h, "z": mean}


def mean_beliefs(rssm, embedded, actions):
    state = {
        "h": torch.zeros(1, rssm.deter_dim, device=embedded.device),
        "z": torch.zeros(1, rssm.stoch_dim, device=embedded.device),
    }
    previous_action = torch.zeros(1, actions.shape[-1], device=embedded.device)
    beliefs = []
    for timestep in range(len(actions)):
        state = observe_mean_step(rssm, embedded[timestep:timestep + 1], previous_action, state)
        beliefs.append(torch.cat([state["h"], state["z"]], dim=-1).squeeze(0))
        previous_action = actions[timestep:timestep + 1]
    return torch.stack(beliefs)


def correlation(pred, target):
    if pred.std() < 1e-8 or target.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(pred, target)[0, 1])


def action_metrics(pred, target):
    per_dimension = [correlation(pred[..., dim].ravel(), target[..., dim].ravel()) for dim in range(pred.shape[-1])]
    valid = [value for value in per_dimension if np.isfinite(value)]
    return {
        "mse": float(np.mean((pred - target) ** 2)),
        "mean_correlation": float(np.mean(valid)) if valid else float("nan"),
        "per_dimension_correlation": per_dimension,
        "gripper_sign_agreement": float(np.mean((pred[..., -1] >= 0) == (target[..., -1] >= 0))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--wm-checkpoint", required=True)
    parser.add_argument("--diff-checkpoint", required=True)
    parser.add_argument("--model-config", default="src/blind_diffusion/configs/model/rssm_small.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--demo-count", type=int, default=40)
    parser.add_argument("--anchors", type=int, nargs="+", default=(0, 25, 50, 100, 200, 400))
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizon, context = 8, 42
    wm_checkpoint = load_checkpoint(args.wm_checkpoint, map_location=device)
    diff_checkpoint = load_checkpoint(args.diff_checkpoint, map_location=device)
    config = OmegaConf.load(args.model_config)
    diff_config = diff_checkpoint["config"]["diffusion"]
    norm = wm_checkpoint["norm"]

    wm = WorldModel(19, 7, config).to(device).eval()
    wm.load_state_dict(wm_checkpoint["model"])
    denoiser = UNet1D(7, diff_config["horizon"], 230, base_ch=diff_config["base_ch"]).to(device)
    diffusion = GaussianDiffusion(denoiser, diff_config["timesteps"], diff_config["schedule"], diff_config["prediction_type"]).to(device).eval()
    diffusion.load_state_dict(diff_checkpoint["model"])
    # Checkpoints store NumPy normalization arrays, which may be float64.
    # Keep the offline observations on the float32 model boundary.
    obs_mean = torch.tensor(norm["obs_mean"], device=device, dtype=torch.float32)
    obs_std = torch.tensor(norm["obs_std"], device=device, dtype=torch.float32)

    pairs = []
    with h5py.File(args.dataset, "r") as handle, torch.no_grad():
        names = sorted(handle["data"].keys())
        indices = np.linspace(0, len(names) - 1, min(args.demo_count, len(names)), dtype=int)
        for demo_index in indices:
            group = handle[f"data/{names[demo_index]}"]
            raw_obs = np.concatenate([group["obs"][key][:].astype(np.float32) for key in OBS_KEYS], axis=-1)
            raw_actions = group["actions"][:].astype(np.float32)
            obs = (torch.tensor(raw_obs, device=device) - obs_mean) / obs_std
            actions = torch.tensor(raw_actions, device=device)
            embedded = wm.encoder(obs)
            full = mean_beliefs(wm.rssm, embedded, actions)
            for anchor in args.anchors:
                if anchor + horizon > len(actions):
                    continue
                start = max(0, anchor - context)
                short = mean_beliefs(wm.rssm, embedded[start:anchor + 1], actions[start:anchor + 1])[-1]
                pairs.append({
                    "demo": names[demo_index],
                    "anchor": anchor,
                    "stateful": full[anchor],
                    "window": short,
                    "target": actions[anchor:anchor + horizon],
                })

    if not pairs:
        raise RuntimeError("No eligible demo/anchor pairs")
    stateful = torch.stack([pair["stateful"] for pair in pairs])
    window = torch.stack([pair["window"] for pair in pairs])
    target = torch.stack([pair["target"] for pair in pairs])
    terminal_noise = torch.randn(target.shape[0], target.shape[2], target.shape[1], device=device)
    with torch.no_grad():
        stateful_actions = diffusion.sample(target.transpose(1, 2).shape, stateful, sampler="ddim", inference_steps=10, initial_noise=terminal_noise).transpose(1, 2)
        window_actions = diffusion.sample(target.transpose(1, 2).shape, window, sampler="ddim", inference_steps=10, initial_noise=terminal_noise).transpose(1, 2)

    anchors = np.asarray([pair["anchor"] for pair in pairs])
    stateful_np, window_np, target_np = (value.cpu().numpy() for value in (stateful_actions, window_actions, target))
    shift = (stateful - window).norm(dim=-1)
    window_norm = window.norm(dim=-1).clamp_min(1e-8)
    metrics = {
        "num_pairs": len(pairs),
        "device": str(device),
        "belief_control": "posterior_mean",
        "window_context_steps": context,
        "all": {
            "condition_l2_mean": float(shift.mean()),
            "condition_l2_median": float(shift.median()),
            "relative_condition_shift_mean": float((shift / window_norm).mean()),
            "stateful": action_metrics(stateful_np, target_np),
            "window": action_metrics(window_np, target_np),
        },
        "by_anchor": {},
        "pairs": [{"demo": pair["demo"], "anchor": pair["anchor"]} for pair in pairs],
    }
    for anchor in sorted(set(anchors.tolist())):
        mask = anchors == anchor
        metrics["by_anchor"][str(anchor)] = {
            "n": int(mask.sum()),
            "condition_l2_mean": float(shift[mask].mean()),
            "relative_condition_shift_mean": float((shift[mask] / window_norm[mask]).mean()),
            "relative_condition_shift_fraction_gt_0_25": float(((shift[mask] / window_norm[mask]) > 0.25).float().mean()),
            "stateful": action_metrics(stateful_np[mask], target_np[mask]),
            "window": action_metrics(window_np[mask], target_np[mask]),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, allow_nan=True) + "\n")
    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
