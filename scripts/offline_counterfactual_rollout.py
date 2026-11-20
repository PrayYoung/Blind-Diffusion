"""Measure causal belief drift from recursively policy-generated actions.

This is an evaluation-only diagnostic. It holds each demonstration's
observations fixed, but compares two posterior RSSM recursions:

* teacher-forced: obs[t] with the demonstration action[t - 1]
* counterfactual: obs[t] with the policy's sampled action[t - 1]

The diffusion policy always samples a fresh DDIM-10 first action from the
counterfactual belief. No optimizer, checkpoint, simulator, or dataset value
is modified.
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


OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]


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


def first_crossing(values: np.ndarray, threshold: float) -> int | None:
    hits = np.flatnonzero(values > threshold)
    return int(hits[0]) if len(hits) else None


def coupled_observe_step(rssm: RSSM, obs_embed: torch.Tensor, left_action: torch.Tensor,
                         left_state: dict, right_action: torch.Tensor, right_state: dict) -> tuple[dict, dict]:
    """Advance two posterior states with identical reparameterization noise.

    At step zero both branches have equal inputs, so their output must be
    exactly equal. Thereafter, paired noise prevents posterior sampling noise
    from being misreported as action-history-caused divergence.
    """
    left_h = rssm.gru(torch.cat([left_state["z"], left_action], dim=-1), left_state["h"])
    right_h = rssm.gru(torch.cat([right_state["z"], right_action], dim=-1), right_state["h"])
    left_mean, left_std = rssm._get_stats(rssm.post_net(torch.cat([left_h, obs_embed], dim=-1)))
    right_mean, right_std = rssm._get_stats(rssm.post_net(torch.cat([right_h, obs_embed], dim=-1)))
    eps = torch.randn_like(left_std)
    return (
        {"h": left_h, "z": left_mean + eps * left_std},
        {"h": right_h, "z": right_mean + eps * right_std},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--wm-checkpoint", required=True)
    parser.add_argument("--diff-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-config", default="src/blind_diffusion/configs/model/rssm_small.yaml")
    parser.add_argument("--demo-start", type=int, default=160)
    parser.add_argument("--num-demos", type=int, default=0, help="0 selects every eligible demo")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sampler-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with h5py.File(args.dataset, "r") as h5:
        candidates = sorted(h5["data"].keys())[args.demo_start:]
        rows = []
        for name in candidates:
            group = h5[f"data/{name}"]
            if len(group["actions"]) < args.steps:
                continue
            actions = group["actions"][: args.steps].astype(np.float32)
            obs = np.concatenate([group["obs"][key][: args.steps].astype(np.float32) for key in OBS_KEYS], axis=-1)
            rows.append((name, obs, actions))
            if args.num_demos and len(rows) >= args.num_demos:
                break

    if not rows:
        raise ValueError(f"No demonstrations at or after index {args.demo_start} contain {args.steps} steps")

    names = [row[0] for row in rows]
    obs = torch.tensor(np.stack([row[1] for row in rows]), device=device)
    demo_actions = torch.tensor(np.stack([row[2] for row in rows]), device=device)

    wm_ckpt = load_checkpoint(args.wm_checkpoint, map_location=device)
    norm = wm_ckpt["norm"]
    obs_mean = torch.tensor(norm["obs_mean"], device=device, dtype=obs.dtype)
    obs_std = torch.tensor(norm["obs_std"], device=device, dtype=obs.dtype)
    obs = (obs - obs_mean) / obs_std

    model_cfg = OmegaConf.load(args.model_config)
    wm = WorldModel(obs.shape[-1], demo_actions.shape[-1], model_cfg).to(device)
    wm.load_state_dict(wm_ckpt["model"])
    wm.eval()

    diff_ckpt = load_checkpoint(args.diff_checkpoint, map_location=device)
    diff_cfg = diff_ckpt["config"]["diffusion"]
    cond_dim = model_cfg.rssm.deter_dim + model_cfg.rssm.stoch_dim
    denoiser = UNet1D(demo_actions.shape[-1], diff_cfg["horizon"], cond_dim, base_ch=diff_cfg["base_ch"]).to(device)
    diffusion = GaussianDiffusion(denoiser, diff_cfg["timesteps"], diff_cfg["schedule"], prediction_type=diff_cfg["prediction_type"]).to(device)
    diffusion.load_state_dict(diff_ckpt["model"])
    diffusion.eval()

    batch = len(rows)
    h_dim = model_cfg.rssm.deter_dim
    zero_h = torch.zeros(batch, h_dim, device=device)
    zero_z = torch.zeros(batch, model_cfg.rssm.stoch_dim, device=device)
    zero_action = torch.zeros(batch, demo_actions.shape[-1], device=device)
    teacher_state = {"h": zero_h.clone(), "z": zero_z.clone()}
    counterfactual_state = {"h": zero_h.clone(), "z": zero_z.clone()}
    counterfactual_prev_action = zero_action.clone()
    trajectory = []

    with torch.no_grad():
        embedded = wm.encoder(obs.reshape(-1, obs.shape[-1])).reshape(batch, args.steps, -1)
        for step in range(args.steps):
            teacher_prev_action = zero_action if step == 0 else demo_actions[:, step - 1]
            teacher_state, counterfactual_state = coupled_observe_step(
                wm.rssm,
                embedded[:, step],
                teacher_prev_action,
                teacher_state,
                counterfactual_prev_action,
                counterfactual_state,
            )
            teacher_belief = torch.cat([teacher_state["h"], teacher_state["z"]], dim=-1)
            counterfactual_belief = torch.cat([counterfactual_state["h"], counterfactual_state["z"]], dim=-1)
            generated = diffusion.sample(
                (batch, demo_actions.shape[-1], diff_cfg["horizon"]),
                counterfactual_belief,
                sampler="ddim",
                inference_steps=args.sampler_steps,
            )[:, :, 0]
            action_error = generated - demo_actions[:, step]
            trajectory.append({
                "step": step,
                "mean_h_l2_difference": float((counterfactual_state["h"] - teacher_state["h"]).norm(dim=-1).mean()),
                "mean_z_l2_difference": float((counterfactual_state["z"] - teacher_state["z"]).norm(dim=-1).mean()),
                "mean_belief_l2_difference": float((counterfactual_belief - teacher_belief).norm(dim=-1).mean()),
                "counterfactual_h_norm_mean": float(counterfactual_state["h"].norm(dim=-1).mean()),
                "counterfactual_z_norm_mean": float(counterfactual_state["z"].norm(dim=-1).mean()),
                "teacher_h_norm_mean": float(teacher_state["h"].norm(dim=-1).mean()),
                "teacher_z_norm_mean": float(teacher_state["z"].norm(dim=-1).mean()),
                "policy_action_mse_to_demo": float(action_error.square().mean()),
                "policy_first_action_l2_to_demo_mean": float(action_error.norm(dim=-1).mean()),
                "policy_action_absmax": float(generated.abs().max()),
                "gripper_sign_agreement": float(((generated[:, -1] >= 0) == (demo_actions[:, step, -1] >= 0)).float().mean()),
            })
            counterfactual_prev_action = generated

    belief_div = np.asarray([row["mean_belief_l2_difference"] for row in trajectory])
    action_mse = np.asarray([row["policy_action_mse_to_demo"] for row in trajectory])
    result = {
        "id": "20260825_offline_counterfactual_rollout",
        "protocol": {
            "training": False,
            "simulator": False,
            "observations": "fixed demonstration observations at every step",
            "teacher_forced_state": "posterior(obs[t], demonstration_action[t-1])",
            "counterfactual_state": "posterior(obs[t], sampled_policy_action[t-1])",
            "posterior_sampling": "paired reparameterization noise across branches at each step",
            "policy": f"DDIM-{args.sampler_steps} first-action RHC sampling from counterfactual posterior belief",
        },
        "inputs": {
            "dataset": args.dataset,
            "wm_checkpoint": args.wm_checkpoint,
            "diff_checkpoint": args.diff_checkpoint,
            "demo_names": names,
            "steps_per_demo": args.steps,
            "seed": args.seed,
        },
        "thresholds": {
            "belief_l2_first_material_divergence": 5.0,
            "belief_l2_severe_divergence": 10.0,
            "policy_action_mse_first_material_divergence": 0.05,
        },
        "summary": {
            "belief_l2_mean_final": float(belief_div[-1]),
            "belief_l2_mean_max": float(belief_div.max()),
            "first_step_belief_l2_gt_5": first_crossing(belief_div, 5.0),
            "first_step_belief_l2_gt_10": first_crossing(belief_div, 10.0),
            "first_step_policy_action_mse_gt_0_05": first_crossing(action_mse, 0.05),
            "counterfactual_action_mse_mean": float(action_mse.mean()),
            "counterfactual_action_mse_final": float(action_mse[-1]),
        },
        "per_step": trajectory,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
