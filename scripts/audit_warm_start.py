"""Probe initial-state prediction using expert actions and fixed visible prefixes."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf
import torch

from blind_diffusion.train.beliefs import compute_pre_action_beliefs
from blind_diffusion.train.train_world_model import WorldModel
from blind_diffusion.utils.checkpoint import load_checkpoint


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    cfg = OmegaConf.create(checkpoint["config"])
    if cfg.normalize_action:
        raise ValueError("This physical startup probe requires native actions")
    norm = checkpoint["norm"]
    mean, std = [torch.as_tensor(norm[key], device=device, dtype=torch.float32)
                 for key in ("obs_mean", "obs_std")]
    totals = {str(prefix): {} for prefix in (5, 10, 20, 0)}
    model = None
    with h5py.File(args.dataset, "r") as dataset:
        for name in checkpoint["split"]["val"]:
            demo = dataset["data"][name]
            length = min(64, len(demo["actions"]))
            obs = torch.from_numpy(np.concatenate([demo["obs"][key][:length] for key in cfg.task.obs_keys], -1)).float().to(device)[None]
            actions = torch.from_numpy(demo["actions"][:length]).float().to(device)[None]
            if model is None:
                model = WorldModel(obs.shape[-1], actions.shape[-1], cfg.model).to(device).eval()
                model.load_state_dict(checkpoint["model"])
            normalized = (obs - mean) / std
            # Stop scoring after a terminal transition, even if the file has a tail.
            dones = np.asarray(demo["dones"][:length]).reshape(-1)
            terminal = np.flatnonzero(dones)
            stop = min(length, int(terminal[0]) + 1) if len(terminal) else length
            for prefix in (5, 10, 20, 0):
                blackout = torch.zeros((1, length), device=device, dtype=torch.bool)
                if prefix:
                    blackout[:, prefix:] = True
                beliefs = compute_pre_action_beliefs(model, normalized, actions, deterministic=True, blackout=blackout)
                prediction = model.obs_head(beliefs)
                raw = prediction * std + mean
                for step in range(stop):
                    row = totals[str(prefix)].setdefault(str(step), {"count": 0, "normalized_mse": 0., "eef_error_m": 0., "object_error_m": 0.})
                    row["count"] += 1
                    row["normalized_mse"] += float((prediction[:, step] - normalized[:, step]).square().mean())
                    offset = 0
                    for key in cfg.task.obs_keys:
                        width = int(demo["obs"][key].shape[-1])
                        if key in ("robot0_eef_pos", "object"):
                            label = "eef_error_m" if key == "robot0_eef_pos" else "object_error_m"
                            row[label] += float((raw[:, step, offset:offset + 3] - obs[:, step, offset:offset + 3]).norm())
                        offset += width
    for rows in totals.values():
        for row in rows.values():
            for key in ("normalized_mse", "eef_error_m", "object_error_m"):
                row[key] /= row["count"]
    result = {"validation_demos": checkpoint["split"]["val"], "by_visible_prefix": totals,
              "zero_prefix_means": "All observations remain visible", "action_history": "Recorded expert actions in every condition",
              "interpretation": "A diagnostic of startup and prior prediction under expert actions, not a control evaluation or a causal attribution to a training bug."}
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({prefix: {step: rows.get(str(step)) for step in (4, 5, 10, 20, 30, 40, 50)}
                      for prefix, rows in totals.items()}, indent=2))


if __name__ == "__main__":
    main()
