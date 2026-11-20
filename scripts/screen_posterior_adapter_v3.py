"""Frozen-WM posterior-mean residual-adapter mechanism screen.

Only the adapter is trainable.  It uses the frozen encoder, deterministic RSSM
state, and frozen posterior mean to produce a gated correction to that mean.
The frozen posterior standard deviation is retained unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from audit_full_checkpoint_imitation import metric_bundle
from blind_diffusion.diffusion.diffusion import GaussianDiffusion
from blind_diffusion.diffusion.model import UNet1D
from blind_diffusion.train.train_world_model import WorldModel
from blind_diffusion.utils.checkpoint import load_checkpoint
from screen_wm_robustness import load_data


class PosteriorMeanAdapter(nn.Module):
    """Small zero-initialized gated residual on the posterior-mean path."""

    def __init__(self, embed_dim: int, deter_dim: int, stoch_dim: int):
        super().__init__()
        in_dim = embed_dim + deter_dim + stoch_dim
        self.residual = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ELU(), nn.Linear(64, stoch_dim)
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.gate = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())

    def forward(self, embed: torch.Tensor, h: torch.Tensor, mean: torch.Tensor, return_gate=False):
        features = torch.cat([embed, h, mean], dim=-1)
        gate = self.gate(features)
        delta = gate * self.residual(features)
        return (delta, gate) if return_gate else delta


def policy_windows(episodes, episode_ids, horizon=50):
    windows = []
    for episode_id in episode_ids:
        trajectory = episodes[episode_id]["trajectory"]
        obs = np.asarray([step["observation"] for step in trajectory], dtype=np.float32)
        actions = np.asarray([step["action"] for step in trajectory], dtype=np.float32)
        windows.extend((obs[start : start + horizon], actions[start : start + horizon])
                       for start in range(0, len(actions) - horizon + 1, 10))
    return windows


def rollout(wm, adapter, obs, actions, stochastic=False, seed=0):
    """Causal pre-action replay with adapter-corrected mean as recurrent z."""
    if stochastic:
        torch.manual_seed(seed)
    batch, length, _ = obs.shape
    embed = wm.encoder(obs.reshape(-1, obs.shape[-1])).reshape(batch, length, -1)
    h = torch.zeros(batch, wm.rssm.deter_dim, device=obs.device)
    z = torch.zeros(batch, wm.rssm.stoch_dim, device=obs.device)
    values = {key: [] for key in ("h", "mean", "z", "prior_mean", "delta", "gate", "posterior_std")}
    for timestep in range(length):
        previous_action = torch.zeros_like(actions[:, 0]) if timestep == 0 else actions[:, timestep - 1]
        h = wm.rssm.gru(torch.cat([z, previous_action], dim=-1), h)
        prior_mean, _ = wm.rssm._get_stats(wm.rssm.prior_net(h))
        base_mean, base_std = wm.rssm._get_stats(wm.rssm.post_net(torch.cat([h, embed[:, timestep]], dim=-1)))
        delta, gate = adapter(embed[:, timestep], h, base_mean, return_gate=True)
        corrected_mean = base_mean + delta
        z = corrected_mean + (torch.randn_like(base_std) * base_std if stochastic else 0.0)
        values["h"].append(h)
        values["mean"].append(corrected_mean)
        values["z"].append(z)
        values["prior_mean"].append(prior_mean)
        values["delta"].append(delta)
        values["gate"].append(gate)
        values["posterior_std"].append(base_std)
    return {"embed": embed, **{key: torch.stack(value, dim=1) for key, value in values.items()}}


def first_above(values, threshold):
    return next((index for index, value in enumerate(values) if value > threshold), None)


@torch.no_grad()
def trajectory_stats(wm, adapter, episodes, mean, std, reference):
    output = []
    for episode_id, episode in enumerate(episodes):
        trajectory = episode["trajectory"]
        raw_obs = torch.tensor(np.asarray([step["observation"] for step in trajectory], dtype=np.float32), device=mean.device)[None]
        actions = torch.tensor(np.asarray([step["action"] for step in trajectory], dtype=np.float32), device=mean.device)[None]
        deterministic = rollout(wm, adapter, (raw_obs - mean) / std, actions)
        sampled = rollout(wm, adapter, (raw_obs - mean) / std, actions, stochastic=True, seed=100 + episode_id)
        norms = {
            "encoder": deterministic["embed"][0].norm(dim=-1).tolist(),
            "posterior_mean": deterministic["mean"][0].norm(dim=-1).tolist(),
            "sampled_z": sampled["z"][0].norm(dim=-1).tolist(),
        }
        output.append({
            "episode": episode_id,
            "posterior_mean_peak": max(norms["posterior_mean"]),
            "sampled_z_peak": max(norms["sampled_z"]),
            "encoder_peak": max(norms["encoder"]),
            "adapter_residual_peak": float(deterministic["delta"][0].norm(dim=-1).max()),
            "adapter_residual_norm_profile": deterministic["delta"][0].norm(dim=-1).tolist(),
            "adapter_gate_min_mean_max": [float(deterministic["gate"][0].min()), float(deterministic["gate"][0].mean()), float(deterministic["gate"][0].max())],
            "first_above_frozen_expert_reference": {
                key: first_above(value, reference[key]) for key, value in norms.items()
            },
        })
    return output


def sample_batch(source, indices, device):
    batch = [source[index] for index in indices]
    return (torch.tensor(np.stack([item[0] for item in batch]), device=device),
            torch.tensor(np.stack([item[1] for item in batch]), device=device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--offpolicy", required=True)
    parser.add_argument("--wm", required=True)
    parser.add_argument("--diff", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full-observation-decode", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    expert_rows, _ = load_data(args.dataset)
    wm_checkpoint = load_checkpoint(args.wm, map_location=device)
    mean = torch.tensor(wm_checkpoint["norm"]["obs_mean"], device=device, dtype=torch.float32)
    std = torch.tensor(wm_checkpoint["norm"]["obs_std"], device=device, dtype=torch.float32)
    model_cfg = OmegaConf.load("src/blind_diffusion/configs/model/rssm_small.yaml")
    wm = WorldModel(19, 7, model_cfg).to(device)
    wm.load_state_dict(wm_checkpoint["model"])
    wm.eval()
    for parameter in wm.parameters():
        parameter.requires_grad_(False)

    episodes = json.loads(Path(args.offpolicy).read_text())["episodes"]
    # Episodes 0--2 are adaptation data. Episode 3 is the preregistered held-out trajectory.
    off_policy = policy_windows(episodes, [0, 1, 2])
    expert = [(obs[start : start + 50], actions[start : start + 50])
              for obs, actions in expert_rows for start in range(len(actions) - 49)]
    rng = np.random.RandomState(0)
    adapter = PosteriorMeanAdapter(model_cfg.encoder.hidden_dim, model_cfg.rssm.deter_dim, model_cfg.rssm.stoch_dim).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-4)

    for _ in range(args.steps):
        off_obs, off_actions = sample_batch(off_policy, rng.randint(len(off_policy), size=16), device)
        expert_obs, expert_actions = sample_batch(expert, rng.randint(len(expert), size=16), device)
        off = rollout(wm, adapter, (off_obs - mean) / std, off_actions)
        clean = rollout(wm, adapter, (expert_obs - mean) / std, expert_actions)
        # Decode real OOD observations and retain action-conditioned recurrent consistency.
        decoded = wm.obs_head(torch.cat([off["h"], off["mean"]], dim=-1))
        off_normalized = (off_obs - mean) / std
        # v3.1 uses the production time burn-in, not an erroneous observation-dimension slice.
        observation_loss = F.mse_loss(decoded[:, 7:], off_normalized[:, 7:]) if args.full_observation_decode else F.mse_loss(decoded[..., 7:], off_normalized[..., 7:])
        transition_loss = F.mse_loss(off["mean"][:, 7:], off["prior_mean"][:, 7:])
        identity_loss = F.mse_loss(clean["delta"][:, 7:], torch.zeros_like(clean["delta"][:, 7:]))
        loss = observation_loss + 0.1 * transition_loss + identity_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Frozen-reference expert statistics and no-adapter control.
    eval_rows = [(obs[:50], actions[:50]) for obs, actions in expert_rows if len(actions) >= 50]
    eval_obs, eval_actions = sample_batch(eval_rows, np.arange(len(eval_rows)), device)
    zero_adapter = PosteriorMeanAdapter(model_cfg.encoder.hidden_dim, model_cfg.rssm.deter_dim, model_cfg.rssm.stoch_dim).to(device)
    with torch.no_grad():
        control = rollout(wm, zero_adapter, (eval_obs - mean) / std, eval_actions)
        adapted = rollout(wm, adapter, (eval_obs - mean) / std, eval_actions)
        sampled_control = rollout(wm, zero_adapter, (eval_obs - mean) / std, eval_actions, stochastic=True, seed=123)
        reference = {
            "encoder": float(control["embed"].norm(dim=-1).max()),
            "posterior_mean": float(control["mean"].norm(dim=-1).max()),
            "sampled_z": float(sampled_control["z"].norm(dim=-1).max()),
        }
        normalized_eval = (eval_obs - mean) / std
        control_reconstruction = F.mse_loss(
            wm.obs_head(torch.cat([control["h"], control["mean"]], dim=-1))[:, 7:], normalized_eval[:, 7:]
        )
        reconstruction = F.mse_loss(
            wm.obs_head(torch.cat([adapted["h"], adapted["mean"]], dim=-1))[:, 7:], normalized_eval[:, 7:]
        )

    diff_checkpoint = load_checkpoint(args.diff, map_location=device)
    diff_cfg = diff_checkpoint["config"]["diffusion"]
    diffusion = GaussianDiffusion(
        UNet1D(7, diff_cfg["horizon"], model_cfg.rssm.deter_dim + model_cfg.rssm.stoch_dim, diff_cfg["base_ch"]).to(device),
        diff_cfg["timesteps"], diff_cfg["schedule"], prediction_type=diff_cfg["prediction_type"],
    ).to(device)
    diffusion.load_state_dict(diff_checkpoint["model"])
    diffusion.eval()
    with torch.no_grad():
        target = eval_actions[:64, :diff_cfg["horizon"]].transpose(1, 2)
        control_condition = torch.cat([control["h"][:64, 0], control["z"][:64, 0]], dim=-1)
        condition = torch.cat([adapted["h"][:64, 0], adapted["z"][:64, 0]], dim=-1)
        torch.manual_seed(123)
        control_prediction = diffusion.sample(target.shape, control_condition, sampler="ddim", inference_steps=10).transpose(1, 2)
        torch.manual_seed(123)
        prediction = diffusion.sample(target.shape, condition, sampler="ddim", inference_steps=10).transpose(1, 2)

    result = {
        "protocol": "policy_induced_posterior_adapter_v3",
        "updates": args.steps,
        "seed": args.seed,
        "ood_observation_decode_scope": "all_observation_dimensions_after_time_burnin" if args.full_observation_decode else "legacy_observation_dimensions_7_plus",
        "adaptation_episodes": [0, 1, 2],
        "held_out_episode": 3,
        "adapter_parameters": sum(parameter.numel() for parameter in adapter.parameters()),
        "frozen_wm_parameters": sum(parameter.numel() for parameter in wm.parameters()),
        "expert_reconstruction_mse": float(reconstruction),
        "expert_reconstruction_control_mse": float(control_reconstruction),
        "expert_h_drift_mse": float(F.mse_loss(adapted["h"], control["h"])),
        "expert_posterior_mean_drift_mse": float(F.mse_loss(adapted["mean"], control["mean"])),
        "expert_imitation": metric_bundle(prediction.cpu().numpy(), target.transpose(1, 2).cpu().numpy()),
        "expert_imitation_control": metric_bundle(control_prediction.cpu().numpy(), target.transpose(1, 2).cpu().numpy()),
        "frozen_expert_reference_max_norms": reference,
        "policy_trajectory_control_stats": trajectory_stats(wm, zero_adapter, episodes, mean, std, reference),
        "policy_trajectory_stats": trajectory_stats(wm, adapter, episodes, mean, std, reference),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
