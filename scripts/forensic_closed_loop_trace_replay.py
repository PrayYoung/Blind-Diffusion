"""Offline forensic replay of the saved closed-loop trace.

The historical trace logs 15 of the 19 low-dimensional observation values:
the four object-quaternion values were not retained. This script never uses a
simulator. It reconstructs all logged components exactly, sweeps every
demonstration initial object quaternion as a transparent nuisance control,
and replays the recorded policy actions through the frozen world model.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from blind_diffusion.models.encoders import MLPEncoder
from blind_diffusion.models.heads import DoneHead, ObsHead, RewardHead
from blind_diffusion.models.rssm import RSSM
from blind_diffusion.utils.checkpoint import load_checkpoint


OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]


class WorldModel(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, cfg_model):
        super().__init__()
        self.encoder = MLPEncoder(obs_dim, cfg_model.encoder.hidden_dim, cfg_model.encoder.layers)
        self.rssm = RSSM(action_dim, cfg_model.encoder.hidden_dim, cfg_model.rssm.deter_dim,
                         cfg_model.rssm.stoch_dim, cfg_model.rssm.hidden_dim,
                         cfg_model.rssm.min_std, cfg_model.rssm.max_std)
        feature_dim = cfg_model.rssm.deter_dim + cfg_model.rssm.stoch_dim
        self.reward_head = RewardHead(feature_dim, cfg_model.heads.reward_hidden)
        self.done_head = DoneHead(feature_dim, cfg_model.heads.done_hidden)
        self.obs_head = ObsHead(feature_dim, obs_dim, cfg_model.heads.obs_hidden)


def observe_mean(rssm, embed, action, state):
    h = rssm.gru(torch.cat([state["z"], action], dim=-1), state["h"])
    mean, std = rssm._get_stats(rssm.post_net(torch.cat([h, embed], dim=-1)))
    return {"h": h, "z": mean}, mean, std


def summary(x):
    x = np.asarray(x, dtype=np.float64)
    return {"min": float(x.min()), "mean": float(x.mean()), "max": float(x.max())}


def first_true(values):
    hits = np.flatnonzero(values)
    return int(hits[0]) if len(hits) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--wm-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-config", default="src/blind_diffusion/configs/model/rssm_small.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with h5py.File(args.dataset, "r") as h5:
        demos = []
        quaternions = []
        for name in sorted(h5["data"]):
            group = h5[f"data/{name}"]
            obs = np.concatenate([group["obs"][key][:].astype(np.float32) for key in OBS_KEYS], axis=-1)
            actions = group["actions"][:].astype(np.float32)
            demos.append((name, obs, actions))
            quaternions.append(obs[0, 12:16])
    candidate_quaternions = np.stack(quaternions)
    raw_demo = np.concatenate([row[1] for row in demos])

    ckpt = load_checkpoint(args.wm_checkpoint, map_location=device)
    obs_mean = np.asarray(ckpt["norm"]["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(ckpt["norm"]["obs_std"], dtype=np.float32)
    model_cfg = OmegaConf.load(args.model_config)
    wm = WorldModel(raw_demo.shape[-1], demos[0][2].shape[-1], model_cfg).to(device)
    wm.load_state_dict(ckpt["model"])
    wm.eval()
    h_dim, z_dim = model_cfg.rssm.deter_dim, model_cfg.rssm.stoch_dim

    # Teacher-forced reference distributions use posterior means so they are
    # directly comparable to the deterministic intervention below.
    ref_encoder_norm, ref_h_norm, ref_mean_norm = [], [], []
    ref_std_min, ref_std_mean, ref_std_max, ref_std_norm = [], [], [], []
    with torch.no_grad():
        for _, raw_obs, demo_actions in demos:
            obs = torch.tensor((raw_obs - obs_mean) / obs_std, device=device).unsqueeze(0)
            actions = torch.tensor(demo_actions, device=device).unsqueeze(0)
            embeds = wm.encoder(obs.reshape(-1, obs.shape[-1])).reshape(1, len(obs[0]), -1)
            state = {"h": torch.zeros(1, h_dim, device=device), "z": torch.zeros(1, z_dim, device=device)}
            zero = torch.zeros_like(actions[:, 0])
            for t in range(actions.shape[1]):
                prev = zero if t == 0 else actions[:, t - 1]
                state, mean, std = observe_mean(wm.rssm, embeds[:, t], prev, state)
                ref_encoder_norm.append(float(embeds[:, t].norm()))
                ref_h_norm.append(float(state["h"].norm()))
                ref_mean_norm.append(float(mean.norm()))
                ref_std_min.append(float(std.min()))
                ref_std_mean.append(float(std.mean()))
                ref_std_max.append(float(std.max()))
                ref_std_norm.append(float(std.norm()))

    trace = json.loads(Path(args.trace).read_text())
    output_episodes = []
    with torch.no_grad():
        for episode in trace["episodes"]:
            rows = episode["trajectory"]
            t_steps = len(rows)
            # Values 12:16 are object quaternion and are absent from trace.
            observed = np.zeros((t_steps, raw_demo.shape[-1]), dtype=np.float32)
            observed_mask = np.ones(raw_demo.shape[-1], dtype=bool)
            observed_mask[12:16] = False
            observed_indices = np.flatnonzero(observed_mask)
            actions = np.asarray([row["action"] for row in rows], dtype=np.float32)
            for t, row in enumerate(rows):
                eef = np.asarray(row["eef_pos"], dtype=np.float32)
                obj = np.asarray(row["object_pos"], dtype=np.float32)
                observed[t, :3] = eef
                observed[t, 3:7] = np.asarray(row["eef_quat"], dtype=np.float32)
                observed[t, 7:9] = np.asarray(row["gripper_qpos"], dtype=np.float32)
                observed[t, 9:12] = obj
                # RoboMimic Lift's final object features are the vector from
                # object to end effector: eef_position - object_position.
                observed[t, 16:19] = eef - obj

            raw_ood = ((observed[:, observed_mask] < raw_demo[:, observed_mask].min(0)) |
                       (observed[:, observed_mask] > raw_demo[:, observed_mask].max(0)))
            raw_ood_steps = raw_ood.any(axis=1)
            raw_z = (observed - obs_mean) / obs_std
            raw_z_absmax = np.abs(raw_z[:, observed_mask]).max(axis=1)
            # Sweep all source-demo initial quaternions. This makes explicit
            # that an exact replay is unavailable from the old trace alone.
            candidates = np.repeat(observed[None], len(candidate_quaternions), axis=0)
            candidates[:, :, 12:16] = candidate_quaternions[:, None, :]
            normalized = torch.tensor((candidates - obs_mean) / obs_std, device=device)
            embeds = wm.encoder(normalized.reshape(-1, normalized.shape[-1])).reshape(len(candidates), t_steps, -1)
            action_t = torch.tensor(actions, device=device).unsqueeze(0).expand(len(candidates), -1, -1)
            zero = torch.zeros(len(candidates), action_t.shape[-1], device=device)
            actual_state = {"h": torch.zeros(len(candidates), h_dim, device=device), "z": torch.zeros(len(candidates), z_dim, device=device)}
            zero_state = {"h": torch.zeros_like(actual_state["h"]), "z": torch.zeros_like(actual_state["z"])}
            per_step = []
            for t in range(t_steps):
                actual_prev = zero if t == 0 else action_t[:, t - 1]
                actual_state, actual_mean, actual_std = observe_mean(wm.rssm, embeds[:, t], actual_prev, actual_state)
                zero_state, zero_mean, zero_std = observe_mean(wm.rssm, embeds[:, t], zero, zero_state)
                actual_h = actual_state["h"].norm(dim=-1).cpu().numpy()
                actual_z = actual_state["z"].norm(dim=-1).cpu().numpy()
                actual_post = actual_mean.norm(dim=-1).cpu().numpy()
                actual_embed = embeds[:, t].norm(dim=-1).cpu().numpy()
                logged = rows[t]
                per_step.append({
                    "step": t,
                    "raw_observation": [float(x) if observed_mask[i] else None for i, x in enumerate(observed[t])],
                    "normalized_observation": [float(x) if observed_mask[i] else None for i, x in enumerate(raw_z[t])],
                    "observed_dimension_mask": observed_mask.tolist(),
                    "raw_observation_any_demo_range_violation": bool(raw_ood_steps[t]),
                    "raw_observation_violation_dimensions": observed_indices[np.flatnonzero(raw_ood[t])].tolist(),
                    "observed_normalized_absmax": float(raw_z_absmax[t]),
                    "policy_first_action": actions[t].tolist(),
                    "gripper_object_distance": float(logged["gripper_object_distance"]),
                    "eef_pos": logged["eef_pos"], "eef_quat": logged["eef_quat"],
                    "gripper_qpos": logged["gripper_qpos"], "object_pos": logged["object_pos"],
                    "logged_h_norm": float(logged["h_norm"]), "logged_z_norm": float(logged["z_norm"]),
                    "actual_action_posterior_mean_replay": {
                        "encoder_norm_candidates": summary(actual_embed), "h_norm_candidates": summary(actual_h),
                        "posterior_mean_norm_candidates": summary(actual_post), "sampled_z_norm_equals_mean_candidates": summary(actual_z),
                        "posterior_std_min_candidates": summary(actual_std.min(dim=-1).values.cpu().numpy()),
                        "posterior_std_mean_candidates": summary(actual_std.mean(dim=-1).cpu().numpy()),
                        "posterior_std_max_candidates": summary(actual_std.max(dim=-1).values.cpu().numpy()),
                        "posterior_std_norm_candidates": summary(actual_std.norm(dim=-1).cpu().numpy()),
                    },
                    "zero_action_posterior_mean_replay": {
                        "h_norm_candidates": summary(zero_state["h"].norm(dim=-1).cpu().numpy()),
                        "posterior_mean_norm_candidates": summary(zero_mean.norm(dim=-1).cpu().numpy()),
                    },
                })
            output_episodes.append({
                "episode": episode["episode"],
                "trace_length": t_steps,
                "first_observed_raw_demo_range_violation": first_true(raw_ood_steps),
                "first_observed_normalized_abs_z_gt_3": first_true(raw_z_absmax > 3),
                "first_logged_z_gt_teacher_forced_max": first_true(np.array([x["z_norm"] for x in rows]) > 11.191254615783691),
                "per_step": per_step,
            })

    result = {
        "id": "20260825_closed_loop_trace_forensics",
        "trace_limitation": "The historical trace omits raw object quaternion dimensions 12:16; exact observation replay is impossible. The replay sweeps all 200 demonstration initial quaternions and reports candidate ranges.",
        "controls": {
            "actual_action_history": "recorded online policy actions, posterior mean intervention",
            "zero_action_history": "all RSSM action inputs zeroed, posterior mean intervention",
            "posterior_sampling": "posterior mean (z=mean) isolates deterministic h/mean dynamics; historical stochastic z cannot be exactly replayed without complete observations and GPU RNG state",
        },
        "teacher_forced_demo_distribution": {
            "raw_observation": {"min": raw_demo.min(0).tolist(), "mean": raw_demo.mean(0).tolist(), "std": raw_demo.std(0).tolist(), "max": raw_demo.max(0).tolist()},
            "encoder_norm": summary(ref_encoder_norm), "h_norm": summary(ref_h_norm),
            "posterior_mean_norm": summary(ref_mean_norm), "posterior_std_min": summary(ref_std_min),
            "posterior_std_mean": summary(ref_std_mean), "posterior_std_max": summary(ref_std_max), "posterior_std_norm": summary(ref_std_norm),
        },
        "episodes": output_episodes,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in ep.items() if k != "per_step"} for ep in output_episodes], indent=2))


if __name__ == "__main__":
    main()
