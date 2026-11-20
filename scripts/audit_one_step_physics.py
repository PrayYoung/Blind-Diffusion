#!/usr/bin/env python3
"""Execute expert and diffusion actions from identical recorded simulator states."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
import robomimic.utils.obs_utils as ObsUtils

from blind_diffusion.diffusion.diffusion import GaussianDiffusion
from blind_diffusion.diffusion.model import UNet1D
from blind_diffusion.env.robomimic_env import make_env
from blind_diffusion.train.beliefs import compute_pre_action_beliefs
from blind_diffusion.train.train_world_model import WorldModel
from blind_diffusion.utils.checkpoint import load_checkpoint


OBS_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")


def flatten(obs):
    return np.concatenate([np.asarray(obs[key], dtype=np.float32) for key in OBS_KEYS])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--wm-checkpoint", required=True)
    parser.add_argument("--diff-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--demo-count", type=int, default=24)
    parser.add_argument("--anchors", type=int, nargs="+", default=(0, 10, 20, 30, 40))
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.wm_checkpoint, map_location=device)
    diff_checkpoint = load_checkpoint(args.diff_checkpoint, map_location=device)
    model_cfg = OmegaConf.load("src/blind_diffusion/configs/model/rssm_small.yaml")
    wm = WorldModel(19, 7, model_cfg).to(device).eval()
    wm.load_state_dict(checkpoint["model"])
    dc = diff_checkpoint["config"]["diffusion"]
    diffusion = GaussianDiffusion(
        UNet1D(7, dc["horizon"], 230, base_ch=dc["base_ch"]).to(device),
        dc["timesteps"],
        dc["schedule"],
        dc["prediction_type"],
    ).to(device).eval()
    diffusion.load_state_dict(diff_checkpoint["model"])
    mean = torch.tensor(checkpoint["norm"]["obs_mean"], device=device, dtype=torch.float32)
    std = torch.tensor(checkpoint["norm"]["obs_std"], device=device, dtype=torch.float32)

    rows = []
    with h5py.File(args.dataset, "r") as handle, torch.no_grad():
        names = sorted(handle["data"].keys())
        for idx in np.linspace(0, len(names) - 1, min(args.demo_count, len(names)), dtype=int):
            demo = handle[f"data/{names[idx]}"]
            actions = demo["actions"][:].astype(np.float32)
            obs = np.concatenate([demo["obs"][key][:].astype(np.float32) for key in OBS_KEYS], axis=-1)
            states = demo["states"][:]
            model_xml = demo.attrs["model_file"]
            normalized = (torch.tensor(obs, device=device) - mean) / std
            beliefs = compute_pre_action_beliefs(wm, normalized.unsqueeze(0), torch.tensor(actions, device=device).unsqueeze(0))[0]
            for anchor in args.anchors:
                if anchor + 1 >= len(actions):
                    continue
                noise = torch.randn(1, 7, dc["horizon"], device=device)
                predicted = diffusion.sample((1, 7, dc["horizon"]), beliefs[anchor:anchor + 1], sampler="ddim", inference_steps=10, initial_noise=noise)[0, :, 0].cpu().numpy()
                rows.append({"demo": names[idx], "anchor": int(anchor), "state": states[anchor], "model": model_xml, "expert": actions[anchor], "predicted": predicted, "recorded_next": obs[anchor + 1]})

    # EnvRobosuite consults this global registry during reset_to. Match the
    # normal low-dimensional evaluator before requesting any observation.
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": list(OBS_KEYS), "rgb": []}})
    env = make_env(args.dataset)
    records = []
    for row in rows:
        state = {"states": row["state"], "model": row["model"]}
        initial = flatten(env.reset_to(state))
        predicted_next, _, _, _ = env.step(row["predicted"])
        predicted_next = flatten(predicted_next)
        # reset_to is required between the paired actions, otherwise expert
        # execution would inherit the predicted action's physical transition.
        env.reset_to(state)
        expert_next, _, _, _ = env.step(row["expert"])
        expert_next = flatten(expert_next)
        recorded_error = float(np.mean((expert_next - row["recorded_next"]) ** 2))
        response = float(np.linalg.norm(expert_next - initial))
        predicted_deviation = float(np.linalg.norm(predicted_next - expert_next))
        records.append({
            "demo": row["demo"], "anchor": row["anchor"],
            "action_l2": float(np.linalg.norm(row["predicted"] - row["expert"])),
            "gripper_sign_match": bool((row["predicted"][-1] >= 0) == (row["expert"][-1] >= 0)),
            "expert_restore_next_obs_mse": recorded_error,
            "expert_response_l2": response,
            "predicted_vs_expert_next_obs_l2": predicted_deviation,
            "response_ratio": predicted_deviation / max(response, 1e-8),
        })
    if hasattr(env, "close"):
        env.close()
    result = {
        "n": len(records), "device": str(device), "records": records,
        "aggregate": {
            "expert_restore_next_obs_mse_mean": float(np.mean([x["expert_restore_next_obs_mse"] for x in records])),
            "action_l2_median": float(np.median([x["action_l2"] for x in records])),
            "gripper_sign_agreement": float(np.mean([x["gripper_sign_match"] for x in records])),
            "predicted_vs_expert_next_obs_l2_median": float(np.median([x["predicted_vs_expert_next_obs_l2"] for x in records])),
            "response_ratio_median": float(np.median([x["response_ratio"] for x in records])),
            "response_ratio_fraction_ge_1": float(np.mean([x["response_ratio"] >= 1 for x in records])),
        },
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
