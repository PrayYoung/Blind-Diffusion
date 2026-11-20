"""Held-out policy-state screen for a structurally separate OOD RSSM.

The expert WM and diffusion checkpoint are read-only baselines.  The OOD model
has independent parameters and learns actual policy-state transitions rather
than an expert-latent target.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from audit_full_checkpoint_imitation import WorldModel
from blind_diffusion.models.encoders import MLPEncoder
from blind_diffusion.models.heads import ObsHead
from blind_diffusion.models.losses import kl_normal
from blind_diffusion.models.rssm import RSSM
from blind_diffusion.utils.checkpoint import load_checkpoint
from screen_wm_robustness import load_data


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_state_dict(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class SeparateOODWorldModel(nn.Module):
    """A small independent observer, posterior, dynamics, and decoder path."""

    def __init__(self):
        super().__init__()
        self.encoder = MLPEncoder(19, 64, 2)
        self.rssm = RSSM(7, 64, 96, 16, 96)
        self.obs_head = ObsHead(112, 19, 128)

    def decode(self, h, z):
        return self.obs_head(torch.cat([h, z], dim=-1))


class FeedForwardPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(26, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 19))

    def forward(self, observation, action):
        return observation + self.net(torch.cat([observation, action], dim=-1))


def policy_sequences(episodes, ids):
    return [(np.asarray([step["observation"] for step in episodes[index]["trajectory"]], np.float32),
             np.asarray([step["action"] for step in episodes[index]["trajectory"]], np.float32))
            for index in ids]


def windows(sequences, horizon=50):
    return [(obs[start : start + horizon], actions[start : start + horizon])
            for obs, actions in sequences
            for start in range(0, len(actions) - horizon + 1, 10)]


def batch(source, indices, device):
    chosen = [source[index] for index in indices]
    return (torch.tensor(np.stack([item[0] for item in chosen]), device=device),
            torch.tensor(np.stack([item[1] for item in chosen]), device=device))


def teacher_roll(model, observations, actions, is_expert=False):
    """Deterministic posterior teacher forcing and actual next-observation prediction."""
    batch_size, length, _ = observations.shape
    embed = model.encoder(observations.reshape(-1, 19)).reshape(batch_size, length, -1)
    h = torch.zeros(batch_size, model.rssm.deter_dim, device=observations.device)
    z = torch.zeros(batch_size, model.rssm.stoch_dim, device=observations.device)
    current, next_predictions, means, hs, post_stds = [], [], [], [], []
    kl_values = []
    for time in range(length):
        previous = torch.zeros_like(actions[:, 0]) if time == 0 else actions[:, time - 1]
        h = model.rssm.gru(torch.cat([z, previous], -1), h)
        prior_mean, prior_std = model.rssm._get_stats(model.rssm.prior_net(h))
        post_mean, post_std = model.rssm._get_stats(model.rssm.post_net(torch.cat([h, embed[:, time]], -1)))
        z = post_mean
        current.append(model.obs_head(torch.cat([h, z], -1)))
        means.append(post_mean); hs.append(h); post_stds.append(post_std)
        kl_values.append(kl_normal(post_mean, post_std, prior_mean, prior_std).mean(-1))
        if time + 1 < length:
            next_h = model.rssm.gru(torch.cat([z, actions[:, time]], -1), h)
            next_mean, _ = model.rssm._get_stats(model.rssm.prior_net(next_h))
            next_predictions.append(model.obs_head(torch.cat([next_h, next_mean], -1)))
    return {
        "current": torch.stack(current, 1),
        "next": torch.stack(next_predictions, 1),
        "mean": torch.stack(means, 1), "h": torch.stack(hs, 1), "std": torch.stack(post_stds, 1),
        "kl": torch.stack(kl_values, 1),
    }


def initial_state(model, observation):
    embed = model.encoder(observation)
    h = torch.zeros(len(observation), model.rssm.deter_dim, device=observation.device)
    z = torch.zeros(len(observation), model.rssm.stoch_dim, device=observation.device)
    post_mean, _ = model.rssm._get_stats(model.rssm.post_net(torch.cat([h, embed], -1)))
    return h, post_mean


@torch.no_grad()
def open_loop(model, observations, actions, horizons):
    results = {}
    for horizon in horizons:
        errors, h_norms, z_norms = [], [], []
        origins = range(0, observations.shape[0] - horizon, 20)
        for origin in origins:
            h, z = initial_state(model, observations[origin : origin + 1])
            predictions = []
            for action in actions[origin : origin + horizon]:
                h = model.rssm.gru(torch.cat([z, action[None]], -1), h)
                z, _ = model.rssm._get_stats(model.rssm.prior_net(h))
                predictions.append(model.obs_head(torch.cat([h, z], -1)))
                h_norms.append(float(h.norm())); z_norms.append(float(z.norm()))
            target = observations[origin + 1 : origin + horizon + 1]
            errors.append(torch.cat(predictions, dim=0).sub(target).square().mean(-1).cpu().tolist())
        errors = np.asarray(errors)
        results[str(horizon)] = {
            "final_step_mse": float(errors[:, -1].mean()),
            "per_step_mse": errors.mean(0).tolist(),
            "max_h_norm": float(max(h_norms)), "max_z_norm": float(max(z_norms)),
            "finite": bool(np.isfinite(errors).all() and np.isfinite(h_norms).all() and np.isfinite(z_norms).all()),
        }
    return results


@torch.no_grad()
def open_loop_ff(model, observations, actions, horizons):
    results = {}
    for horizon in horizons:
        errors = []
        for origin in range(0, observations.shape[0] - horizon, 20):
            prediction = observations[origin : origin + 1]
            sequence = []
            for action in actions[origin : origin + horizon]:
                prediction = model(prediction, action[None]); sequence.append(prediction)
            target = observations[origin + 1 : origin + horizon + 1]
            errors.append(torch.cat(sequence, dim=0).sub(target).square().mean(-1).cpu().tolist())
        errors = np.asarray(errors)
        results[str(horizon)] = {"final_step_mse": float(errors[:, -1].mean()), "per_step_mse": errors.mean(0).tolist(), "finite": bool(np.isfinite(errors).all())}
    return results


def persistence_open_loop(observations, horizons):
    results = {}
    for horizon in horizons:
        errors = []
        for origin in range(0, observations.shape[0] - horizon, 20):
            target = observations[origin + 1 : origin + horizon + 1]
            errors.append(observations[origin : origin + 1].expand_as(target).sub(target).square().mean(-1).cpu().tolist())
        errors = np.asarray(errors)
        results[str(horizon)] = {"final_step_mse": float(errors[:, -1].mean()), "per_step_mse": errors.mean(0).tolist(), "finite": bool(np.isfinite(errors).all())}
    return results


def support_gate(expert_rows, policy_episodes, mean, std):
    # Calibrate support using the first 160 demonstrations; measure false positives on the remaining 40.
    calibration = np.concatenate([obs for obs, _ in expert_rows[:160]], 0)
    evaluation = np.concatenate([obs for obs, _ in expert_rows[160:]], 0)
    lo, hi = calibration.min(0), calibration.max(0)
    expert_fpr = float(((evaluation < lo) | (evaluation > hi)).any(1).mean())
    all_reference = np.concatenate([obs for obs, _ in expert_rows], 0)
    all_lo, all_hi = all_reference.min(0), all_reference.max(0)
    records = []
    for episode_id, episode in enumerate(policy_episodes):
        observation = np.asarray([step["observation"] for step in episode["trajectory"]], np.float32)
        activation = np.flatnonzero(((observation < lo) | (observation > hi)).any(1))
        known = np.flatnonzero(((observation < all_lo) | (observation > all_hi)).any(1))
        records.append({"episode": episode_id, "gate_first_activation": int(activation[0]) if len(activation) else None,
                        "known_raw_ood_onset": int(known[0]) if len(known) else None})
    return {"branch_rule": "use_ood_if_any_raw_dimension_outside_calibration_demo_support", "expert_false_positive_rate": expert_fpr, "policy_activation": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True); parser.add_argument("--offpolicy", required=True)
    parser.add_argument("--wm", required=True); parser.add_argument("--diff", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000); parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(); device = torch.device("cuda"); torch.manual_seed(args.seed); rng = np.random.RandomState(args.seed)
    expert_rows, _ = load_data(args.dataset)
    checkpoint = load_checkpoint(args.wm, map_location=device)
    mean = torch.tensor(checkpoint["norm"]["obs_mean"], device=device, dtype=torch.float32)
    std = torch.tensor(checkpoint["norm"]["obs_std"], device=device, dtype=torch.float32)
    model_cfg = OmegaConf.load("src/blind_diffusion/configs/model/rssm_small.yaml")
    expert = WorldModel(19, 7, model_cfg).to(device); expert.load_state_dict(checkpoint["model"]); expert.eval()
    for parameter in expert.parameters(): parameter.requires_grad_(False)
    expert_digest_before = digest_state_dict(expert); diffusion_digest_before = digest_file(args.diff)
    episodes = json.loads(Path(args.offpolicy).read_text())["episodes"]
    train_sequences = policy_sequences(episodes, [0, 1, 2]); train_windows = windows(train_sequences)
    branch = SeparateOODWorldModel().to(device); optimizer = torch.optim.Adam(branch.parameters(), lr=1e-3)
    for _ in range(args.steps):
        raw_obs, actions = batch(train_windows, rng.randint(len(train_windows), size=32), device); observations = (raw_obs - mean) / std
        rollout = teacher_roll(branch, observations, actions)
        loss = F.mse_loss(rollout["current"], observations) + F.mse_loss(rollout["next"], observations[:, 1:]) + 0.01 * rollout["kl"].mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    ff = FeedForwardPredictor().to(device); ff_optimizer = torch.optim.Adam(ff.parameters(), lr=1e-3)
    transitions = [(obs[:-1], action[:-1], obs[1:]) for obs, action in train_sequences]
    train_obs = np.concatenate([item[0] for item in transitions]); train_actions = np.concatenate([item[1] for item in transitions]); train_next = np.concatenate([item[2] for item in transitions])
    for _ in range(args.steps // 2):
        indices = rng.randint(len(train_obs), size=128)
        obs = (torch.tensor(train_obs[indices], device=device) - mean) / std; action = torch.tensor(train_actions[indices], device=device); target = (torch.tensor(train_next[indices], device=device) - mean) / std
        loss = F.mse_loss(ff(obs, action), target); ff_optimizer.zero_grad(); loss.backward(); ff_optimizer.step()
    branch.eval(); ff.eval()
    held_obs_raw, held_actions = policy_sequences(episodes, [3])[0]
    held_obs = (torch.tensor(held_obs_raw, device=device) - mean) / std; held_actions = torch.tensor(held_actions, device=device)
    with torch.no_grad():
        held_batch, held_action_batch = held_obs[None], held_actions[None]
        branch_teacher = teacher_roll(branch, held_batch, held_action_batch)
        expert_teacher = teacher_roll(expert, held_batch, held_action_batch, is_expert=True)
        target_next = held_obs[1:]
        ff_one_step = ff(held_obs[:-1], held_actions[:-1])
        one_step = {
            "separate_ood_branch_mse": float(F.mse_loss(branch_teacher["next"][0], target_next)),
            "frozen_expert_wm_mse": float(F.mse_loss(expert_teacher["next"][0], target_next)),
            "persistence_mse": float(F.mse_loss(held_obs[:-1], target_next)),
            "action_conditioned_feedforward_mse": float(F.mse_loss(ff_one_step, target_next)),
        }
        horizons = [5, 10, 20]
        open_loop_metrics = {"separate_ood_branch": open_loop(branch, held_obs, held_actions, horizons),
                     "frozen_expert_wm": open_loop(expert, held_obs, held_actions, horizons),
                     "persistence": persistence_open_loop(held_obs, horizons),
                     "action_conditioned_feedforward": open_loop_ff(ff, held_obs, held_actions, horizons)}
        branch_stability = {"teacher_forced_max_h_norm": float(branch_teacher["h"].norm(dim=-1).max()),
                            "teacher_forced_max_posterior_mean_norm": float(branch_teacher["mean"].norm(dim=-1).max()),
                            "teacher_forced_max_posterior_std": float(branch_teacher["std"].max()),
                            "finite": bool(torch.isfinite(branch_teacher["h"]).all() and torch.isfinite(branch_teacher["mean"]).all())}
    result = {
        "protocol": "separate_ood_world_model_v4", "seed": args.seed, "updates": args.steps,
        "adaptation_episodes": [0, 1, 2], "held_out_episode": 3,
        "separate_branch_parameters": sum(parameter.numel() for parameter in branch.parameters()),
        "frozen_expert_wm_parameters": sum(parameter.numel() for parameter in expert.parameters()),
        "expert_branch_bitwise_unchanged": expert_digest_before == digest_state_dict(expert),
        "diffusion_checkpoint_unchanged": diffusion_digest_before == digest_file(args.diff),
        "held_out_one_step_normalized_mse": one_step, "held_out_open_loop": open_loop_metrics,
        "separate_branch_stability": branch_stability,
        "gate": support_gate(expert_rows, episodes, mean.cpu().numpy(), std.cpu().numpy()),
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
