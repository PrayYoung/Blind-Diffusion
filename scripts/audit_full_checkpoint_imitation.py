"""Offline causal-imitation audit for an already trained WM + diffusion pair.

The full training protocol does not reserve demos, so ``--demo-start`` defines
a post-hoc diagnostic subset rather than a statistically held-out training set.
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
from blind_diffusion.train.beliefs import compute_pre_action_beliefs
from blind_diffusion.utils.checkpoint import load_checkpoint


OBS_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object",
]


class WorldModel(torch.nn.Module):
    """Checkpoint-compatible WM definition without the Hydra trainer import."""

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
        feat_dim = cfg_model.rssm.deter_dim + cfg_model.rssm.stoch_dim
        self.reward_head = RewardHead(feat_dim, cfg_model.heads.reward_hidden)
        self.done_head = DoneHead(feat_dim, cfg_model.heads.done_hidden)
        self.obs_head = ObsHead(feat_dim, obs_dim, cfg_model.heads.obs_hidden)


def correlation(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.std() < 1e-8 or target.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(pred, target)[0, 1])


def metric_bundle(pred: np.ndarray, target: np.ndarray) -> dict:
    # Inputs are [N, H, A].
    per_dim_corr = [correlation(pred[..., dim].ravel(), target[..., dim].ravel()) for dim in range(pred.shape[-1])]
    valid = [x for x in per_dim_corr if np.isfinite(x)]
    return {
        "mse": float(np.mean((pred - target) ** 2)),
        "mean_correlation": float(np.mean(valid)) if valid else float("nan"),
        "per_dimension_correlation": per_dim_corr,
        "pred_range_per_dimension": [[float(pred[..., dim].min()), float(pred[..., dim].max())] for dim in range(pred.shape[-1])],
        "demo_range_per_dimension": [[float(target[..., dim].min()), float(target[..., dim].max())] for dim in range(target.shape[-1])],
        "pred_mean_per_dimension": [float(pred[..., dim].mean()) for dim in range(pred.shape[-1])],
        "pred_std_per_dimension": [float(pred[..., dim].std()) for dim in range(pred.shape[-1])],
        "demo_mean_per_dimension": [float(target[..., dim].mean()) for dim in range(target.shape[-1])],
        "demo_std_per_dimension": [float(target[..., dim].std()) for dim in range(target.shape[-1])],
        "action_absmax": float(np.abs(pred).max()),
        "gripper_sign_agreement": float(np.mean((pred[..., -1] >= 0) == (target[..., -1] >= 0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--wm-checkpoint", required=True)
    parser.add_argument("--diff-checkpoint", required=True)
    parser.add_argument("--model-config", default="src/blind_diffusion/configs/model/rssm_small.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--demo-start", type=int, default=160)
    parser.add_argument("--chunks-per-demo", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizon = 8

    with h5py.File(args.dataset, "r") as h5:
        names = sorted(h5["data"].keys())[args.demo_start:]
        rows, positions = [], []
        for name in names:
            group = h5[f"data/{name}"]
            actions = group["actions"][:].astype(np.float32)
            obs = np.concatenate([group["obs"][key][:].astype(np.float32) for key in OBS_KEYS], axis=-1)
            if len(actions) < 50:
                continue
            max_start = len(actions) - 50
            starts = np.linspace(0, max_start, args.chunks_per_demo, dtype=int)
            t0_values = np.linspace(0, 42, args.chunks_per_demo, dtype=int)
            for start, t0 in zip(starts, t0_values):
                # Both the 50-step window start and its causal action index
                # vary, matching the trainer's possible global positions.
                rows.append((obs[start : start + 50], actions[start : start + 50], t0))
                positions.append(float(start + t0) / max(1, len(actions) - horizon))

    obs = torch.tensor(np.stack([x[0] for x in rows]), device=device)
    actions = torch.tensor(np.stack([x[1] for x in rows]), device=device)
    t0 = torch.tensor([x[2] for x in rows], device=device)

    wm_ckpt = load_checkpoint(args.wm_checkpoint, map_location=device)
    norm = wm_ckpt["norm"]
    obs_mean = torch.tensor(norm["obs_mean"], device=device, dtype=obs.dtype)
    obs_std = torch.tensor(norm["obs_std"], device=device, dtype=obs.dtype)
    obs_normalized = (obs - obs_mean) / obs_std

    model_cfg = OmegaConf.load(args.model_config)
    wm = WorldModel(obs.shape[-1], actions.shape[-1], model_cfg).to(device)
    wm.load_state_dict(wm_ckpt["model"])
    wm.eval()

    diff_ckpt = load_checkpoint(args.diff_checkpoint, map_location=device)
    diff_cfg = diff_ckpt["config"]["diffusion"]
    prediction_type = diff_cfg["prediction_type"]
    cond_dim = model_cfg.rssm.deter_dim + model_cfg.rssm.stoch_dim
    denoiser = UNet1D(actions.shape[-1], diff_cfg["horizon"], cond_dim, base_ch=diff_cfg["base_ch"]).to(device)
    diffusion = GaussianDiffusion(
        denoiser,
        timesteps=diff_cfg["timesteps"],
        schedule=diff_cfg["schedule"],
        prediction_type=prediction_type,
    ).to(device)
    diffusion.load_state_dict(diff_ckpt["model"])
    diffusion.eval()

    with torch.no_grad():
        beliefs = compute_pre_action_beliefs(wm, obs_normalized, actions)
        cond = beliefs[torch.arange(len(rows), device=device), t0]
        target = torch.stack([actions[i, t0[i] : t0[i] + horizon] for i in range(len(rows))])
        sampled = diffusion.sample(target.transpose(1, 2).shape, cond, sampler="ddim", inference_steps=10).transpose(1, 2)

    sampled_np, target_np = sampled.cpu().numpy(), target.cpu().numpy()
    first = metric_bundle(sampled_np[:, :1], target_np[:, :1])
    full = metric_bundle(sampled_np, target_np)
    mean_action = np.broadcast_to(target_np.mean(axis=(0, 1), keepdims=True), target_np.shape)
    zero_action = np.zeros_like(target_np)
    position = np.asarray(positions)
    bins = {}
    for label, lo, hi in (("early", 0.0, 1 / 3), ("middle", 1 / 3, 2 / 3), ("late", 2 / 3, 1.000001)):
        mask = (position >= lo) & (position < hi)
        if mask.any():
            bins[label] = metric_bundle(sampled_np[mask], target_np[mask])

    result = {
        "checkpoint_mode": "exact_full_seed0",
        "post_hoc_demo_subset": {"demo_start": args.demo_start, "demo_count": len(names), "note": "Full training used all demos; this subset was not withheld during optimization."},
        "wm_checkpoint": args.wm_checkpoint,
        "diff_checkpoint": args.diff_checkpoint,
        "prediction_type": prediction_type,
        "normalization": {"obs": "checkpoint obs_mean/obs_std", "actions": "native, unnormalized"},
        "causal_conditioning": "state[t] = posterior(obs[t], action[t-1])",
        "sampler": "ddim",
        "inference_steps": 10,
        "num_chunks": len(rows),
        "teacher_forced_belief_norms": {
            "h": {"mean": float(cond[:, :model_cfg.rssm.deter_dim].norm(dim=-1).mean()), "std": float(cond[:, :model_cfg.rssm.deter_dim].norm(dim=-1).std()), "min": float(cond[:, :model_cfg.rssm.deter_dim].norm(dim=-1).min()), "max": float(cond[:, :model_cfg.rssm.deter_dim].norm(dim=-1).max())},
            "z": {"mean": float(cond[:, model_cfg.rssm.deter_dim:].norm(dim=-1).mean()), "std": float(cond[:, model_cfg.rssm.deter_dim:].norm(dim=-1).std()), "min": float(cond[:, model_cfg.rssm.deter_dim:].norm(dim=-1).min()), "max": float(cond[:, model_cfg.rssm.deter_dim:].norm(dim=-1).max())},
        },
        "full_chunk": full,
        "first_action": first,
        "reference_baselines": {
            "global_mean_action_mse": float(np.mean((mean_action - target_np) ** 2)),
            "zero_action_mse": float(np.mean((zero_action - target_np) ** 2)),
            "ddim_improvement_over_global_mean": float(1.0 - full["mse"] / np.mean((mean_action - target_np) ** 2)),
            "ddim_improvement_over_zero": float(1.0 - full["mse"] / np.mean((zero_action - target_np) ** 2)),
        },
        "position_bins": bins,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
