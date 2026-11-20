"""Frozen-WM OOD-mode audit for the policy-induced adapter experiments."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from blind_diffusion.train.train_world_model import WorldModel
from blind_diffusion.utils.checkpoint import load_checkpoint
from screen_wm_robustness import load_data


OBSERVATION_DIMENSIONS = [
    "eef_pos_x", "eef_pos_y", "eef_pos_z", "eef_quat_w", "eef_quat_x", "eef_quat_y", "eef_quat_z",
    "gripper_qpos_0", "gripper_qpos_1", "object_pos_x", "object_pos_y", "object_pos_z",
    "object_quat_w", "object_quat_x", "object_quat_y", "object_quat_z", "eef_to_object_x", "eef_to_object_y", "eef_to_object_z",
]


def first_crossing(values, threshold):
    return next((index for index, value in enumerate(values) if value > threshold), None)


@torch.no_grad()
def wm_norms(wm, normalized_obs, actions):
    embed = wm.encoder(normalized_obs)
    h = torch.zeros(normalized_obs.shape[0], wm.rssm.deter_dim, device=normalized_obs.device)
    z = torch.zeros(normalized_obs.shape[0], wm.rssm.stoch_dim, device=normalized_obs.device)
    result = {key: [] for key in ("encoder", "h", "posterior_mean", "posterior_std_mean", "posterior_std_max")}
    for timestep in range(normalized_obs.shape[1]):
        previous_action = torch.zeros_like(actions[:, 0]) if timestep == 0 else actions[:, timestep - 1]
        h = wm.rssm.gru(torch.cat([z, previous_action], dim=-1), h)
        mean, std = wm.rssm._get_stats(wm.rssm.post_net(torch.cat([h, embed[:, timestep]], dim=-1)))
        z = mean
        result["encoder"].append(float(embed[:, timestep].norm(dim=-1).max()))
        result["h"].append(float(h.norm(dim=-1).max()))
        result["posterior_mean"].append(float(mean.norm(dim=-1).max()))
        result["posterior_std_mean"].append(float(std.mean()))
        result["posterior_std_max"].append(float(std.max()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--offpolicy", required=True)
    parser.add_argument("--wm", required=True)
    parser.add_argument("--v3-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    expert_rows, _ = load_data(args.dataset)
    expert_obs = np.concatenate([obs for obs, _ in expert_rows], axis=0)
    expert_min = expert_obs.min(axis=0)
    expert_max = expert_obs.max(axis=0)
    checkpoint = load_checkpoint(args.wm, map_location=device)
    mean = np.asarray(checkpoint["norm"]["obs_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["norm"]["obs_std"], dtype=np.float32)
    cfg = OmegaConf.load("src/blind_diffusion/configs/model/rssm_small.yaml")
    wm = WorldModel(19, 7, cfg).to(device)
    wm.load_state_dict(checkpoint["model"])
    wm.eval()
    expert_norm = (expert_obs - mean) / std
    reference = {
        "encoder": None,
        "h": None,
        "posterior_mean": None,
        "posterior_std_mean": None,
        "posterior_std_max": None,
    }
    # The existing v3 uses first 50 expert steps per demonstration for its reference.
    reference_rows = [(obs[:50], act[:50]) for obs, act in expert_rows if len(obs) >= 50]
    sample_obs = torch.tensor(np.stack([obs for obs, _ in reference_rows]), device=device)
    sample_actions = torch.tensor(np.stack([act for _, act in reference_rows]), device=device)
    expert_roll = wm_norms(wm, (sample_obs - torch.tensor(mean, device=device)) / torch.tensor(std, device=device), sample_actions)
    for key, values in expert_roll.items():
        reference[key] = max(values)

    v3 = json.loads(Path(args.v3_result).read_text())
    episodes = json.loads(Path(args.offpolicy).read_text())["episodes"]
    output = []
    for episode_id, episode in enumerate(episodes):
        trajectory = episode["trajectory"]
        obs = np.asarray([step["observation"] for step in trajectory], dtype=np.float32)
        actions = np.asarray([step["action"] for step in trajectory], dtype=np.float32)
        z_scores = np.abs((obs - mean) / std)
        outside_demo = (obs < expert_min) | (obs > expert_max)
        first_outside = np.flatnonzero(outside_demo.any(axis=1))
        onset = int(first_outside[0]) if len(first_outside) else None
        top = []
        if onset is not None:
            ranked = np.argsort(z_scores[onset])[::-1][:6]
            top = [{"dimension": OBSERVATION_DIMENSIONS[index], "z_score": float(z_scores[onset, index]), "value": float(obs[onset, index]), "expert_range": [float(expert_min[index]), float(expert_max[index])]} for index in ranked]
        norms = wm_norms(
            wm,
            (torch.tensor(obs, device=device)[None] - torch.tensor(mean, device=device)) / torch.tensor(std, device=device),
            torch.tensor(actions, device=device)[None],
        )
        control = v3["policy_trajectory_control_stats"][episode_id]
        adapted = v3["policy_trajectory_stats"][episode_id]
        output.append({
            "episode": episode_id,
            "first_raw_observation_outside_demo_range": onset,
            "top_normalized_dimensions_at_onset": top,
            "max_abs_z_per_dimension": {OBSERVATION_DIMENSIONS[index]: float(z_scores[:, index].max()) for index in range(obs.shape[1])},
            "frozen_wm_first_crossing": {key: first_crossing(values, reference[key]) for key, values in norms.items()},
            "frozen_wm_at_v3_adapter_effectiveness_limit": {
                "definition": "the v3 adapted posterior-mean first crossing, or its peak if no crossing",
                "timestep": adapted["first_above_frozen_expert_reference"]["posterior_mean"] if adapted["first_above_frozen_expert_reference"]["posterior_mean"] is not None else int(np.argmax(norms["posterior_mean"])),
                "encoder_norm": None,
                "h_norm": None,
                "posterior_mean_norm": None,
                "posterior_std_mean": None,
                "posterior_std_max": None,
            },
            "v3": {"control": control, "adapted": adapted},
        })
        at = output[-1]["frozen_wm_at_v3_adapter_effectiveness_limit"]["timestep"]
        for key in ("encoder", "h", "posterior_mean", "posterior_std_mean", "posterior_std_max"):
            output[-1]["frozen_wm_at_v3_adapter_effectiveness_limit"][f"{key}_norm" if key in ("encoder", "h", "posterior_mean") else key] = norms[key][at]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"expert_reference": reference, "episodes": output}, indent=2))


if __name__ == "__main__":
    main()
