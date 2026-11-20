"""Audit deterministic held-out prior predictions by rollout offset."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from blind_diffusion.data.robomimic_dataset import build_episode_datasets
from blind_diffusion.train.train_world_model import WorldModel
from blind_diffusion.train.beliefs import previous_actions
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
    _, validation, split = build_episode_datasets(args.dataset, cfg.task.obs_keys, cfg.seq_len,
        cfg.burn_in, cfg.normalize_obs, cfg.normalize_action, cfg.pad_end,
        norm_stats=checkpoint["norm"], split=checkpoint["split"])
    sample = validation[0]
    model = WorldModel(sample["obs"].shape[-1], sample["actions"].shape[-1], cfg.model).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    horizon = 20
    starts = cfg.seq_len - horizon - cfg.burn_in
    if starts < 1:
        raise ValueError("Audit requires at least 20 future observations after burn-in")
    totals = {h: {"squared_error": 0., "persistence_squared_error": 0., "dimensions": 0, "states": 0}
              for h in (1, 5, 10, 20)}
    posterior_error, posterior_dims = 0., 0
    for batch in DataLoader(validation, batch_size=64, shuffle=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        obs, actions, valid, dones = (batch[k] for k in ("obs", "actions", "valid", "dones"))
        post = model.rssm.observe(model.encoder(obs), previous_actions(actions), deterministic=True)
        reconstruction = model.obs_head(torch.cat([post["h"], post["z"]], -1))
        mask = valid[:, cfg.burn_in + 1:]
        posterior_error += float(((reconstruction[:, cfg.burn_in + 1:] - obs[:, cfg.burn_in + 1:]).square() * mask[..., None]).sum())
        posterior_dims += int(mask.sum()) * obs.shape[-1]
        state = {key: post[key][:, cfg.burn_in:cfg.burn_in + starts].reshape(-1, post[key].shape[-1]) for key in ("h", "z")}
        alive = valid[:, cfg.burn_in:cfg.burn_in + starts].clone()
        initial = obs[:, cfg.burn_in:cfg.burn_in + starts]
        for offset in range(1, horizon + 1):
            action = actions[:, cfg.burn_in + offset - 1:cfg.burn_in + offset - 1 + starts]
            alive &= valid[:, cfg.burn_in + offset:cfg.burn_in + offset + starts] & ~dones[:, cfg.burn_in + offset - 1:cfg.burn_in + offset - 1 + starts].bool()
            state = model.rssm.imagine_step(action.reshape(-1, actions.shape[-1]), state, deterministic=True)
            if offset not in totals:
                continue
            target = obs[:, cfg.burn_in + offset:cfg.burn_in + offset + starts]
            pred = model.obs_head(torch.cat([state["h"], state["z"]], -1)).reshape_as(target)
            row = totals[offset]
            row["squared_error"] += float(((pred - target).square() * alive[..., None]).sum())
            row["persistence_squared_error"] += float(((initial - target).square() * alive[..., None]).sum())
            row["states"] += int(alive.sum())
            row["dimensions"] += int(alive.sum()) * obs.shape[-1]
    for row in totals.values():
        row["mse"] = row["squared_error"] / row["dimensions"] if row["dimensions"] else None
        row["persistence_mse"] = row["persistence_squared_error"] / row["dimensions"] if row["dimensions"] else None
    result = {"device": str(device), "validation_demos": split["val"], "validation_windows": len(validation),
        "posterior_mse": posterior_error / posterior_dims, "by_offset": totals,
        "final20_gate_pass": totals[20]["mse"] is not None and totals[20]["mse"] <= .20}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
