"""Separate held-out policy learning from few-step Euler integration error."""
import argparse
import json
from pathlib import Path
import platform

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from blind_diffusion.data.robomimic_dataset import build_episode_datasets
from blind_diffusion.diffusion.policy import build_action_policy
from blind_diffusion.train.beliefs import compute_pre_action_beliefs, sample_conditioning_queries
from blind_diffusion.train.train_world_model import WorldModel
from blind_diffusion.utils.checkpoint import load_checkpoint


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--world-model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(34200)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    wm_checkpoint = load_checkpoint(args.world_model, map_location=device)
    cfg = OmegaConf.create(checkpoint["config"])
    if cfg.policy_type != "flow_matching" or checkpoint["split"] != wm_checkpoint["split"]:
        raise ValueError("Audit requires matching flow/WM episode splits")
    _, validation, split = build_episode_datasets(args.dataset, cfg.task.obs_keys, cfg.seq_len,
        cfg.burn_in, cfg.normalize_obs, cfg.normalize_action, cfg.pad_end,
        norm_stats=wm_checkpoint["norm"], split=wm_checkpoint["split"])
    sample = validation[0]
    obs_dim, act_dim = sample["obs"].shape[-1], sample["actions"].shape[-1]
    wm = WorldModel(obs_dim, act_dim, cfg.model).to(device).eval()
    wm.load_state_dict(wm_checkpoint["model"])
    policy = build_action_policy(cfg, act_dim, cfg.model.rssm.deter_dim + cfg.model.rssm.stoch_dim).to(device).eval()
    policy.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
    totals = {(mode, steps): {"sse": 0., "dimensions": 0, "raw_absmax": 0., "outside": 0,
                             "raw_sse": 0., "first_pred": [], "first_target": []}
              for mode in ("clean", "prior") for steps in (4, 32)}
    depths = []
    belief_norms = {mode: [] for mode in ("clean", "prior")}
    for batch in DataLoader(validation, batch_size=64, shuffle=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        obs, actions, valid = batch["obs"], batch["actions"], batch["valid"]
        queries, blackout = sample_conditioning_queries(valid, batch["is_start"], cfg.diffusion.horizon,
            cfg.burn_in, 1.0, cfg.warmup_steps, cfg.blackout_max_duration)
        depths.extend(blackout.sum(1).cpu().tolist())
        indices = queries[:, None] + torch.arange(cfg.diffusion.horizon, device=device)
        target = actions.gather(1, indices[..., None].expand(-1, -1, act_dim)).transpose(1, 2)
        mask = valid.gather(1, indices)[:, None].expand_as(target)
        conditions = {mode: compute_pre_action_beliefs(wm, obs, actions, deterministic=True,
            blackout=blackout if mode == "prior" else None)[torch.arange(len(obs), device=device), queries]
            for mode in ("clean", "prior")}
        for mode, condition in conditions.items():
            belief_norms[mode].append(condition.cpu())
        for _ in range(4):
            noise = torch.randn_like(target)
            for (mode, steps), row in totals.items():
                raw = policy.sample(target.shape, conditions[mode], inference_steps=steps,
                                    initial_noise=noise, clamp_final=False)
                pred = raw.clamp(-1, 1)
                row["sse"] += float(((pred - target).square() * mask).sum())
                row["raw_sse"] += float(((raw - target).square() * mask).sum())
                row["dimensions"] += int(mask.sum())
                row["outside"] += int(((raw.abs() > 1) & mask).sum())
                row["raw_absmax"] = max(row["raw_absmax"], float(raw[mask].abs().max()))
                row["first_pred"].append(pred[:, :, 0].cpu())
                row["first_target"].append(target[:, :, 0].cpu())
    result = {}
    for (mode, steps), row in totals.items():
        pred, target = (torch.cat(row.pop(key)) for key in ("first_pred", "first_target"))
        pc, tc = pred - pred.mean(0), target - target.mean(0)
        corr = (pc * tc).sum(0) / (pc.square().sum(0) * tc.square().sum(0)).sqrt().clamp_min(1e-12)
        result[f"{mode}_euler{steps}"] = {"chunk_mse": row["sse"] / row["dimensions"],
            "raw_chunk_mse": row["raw_sse"] / row["dimensions"], "first_action_mse": float((pred-target).square().mean()),
            "first_action_correlation_by_dimension": corr.tolist(),
            "gripper_sign_agreement": float((pred[:, -1].sign() == target[:, -1].sign()).float().mean()),
            "raw_absmax": row["raw_absmax"], "raw_outside_fraction": row["outside"] / row["dimensions"]}
    result.update(validation_demos=split["val"], validation_windows=len(validation),
                  prior_depth_mean=sum(depths)/len(depths), prior_depth_max=max(depths),
                  runtime={"python": platform.python_version(), "torch": torch.__version__,
                           "cuda": torch.version.cuda, "device": str(device),
                           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
                  interpretation="Diagnostic only: same queries and four paired noise draws for both conditions and solvers; teacher-forced expert histories do not establish control.")
    result["belief_norm_reference"] = {}
    for mode, values in belief_norms.items():
        belief = torch.cat(values)
        parts = {"h": belief[:, :cfg.model.rssm.deter_dim], "z": belief[:, cfg.model.rssm.deter_dim:]}
        result["belief_norm_reference"][mode] = {
            key: {"mean": float(value.norm(dim=-1).mean()), "max": float(value.norm(dim=-1).max())}
            for key, value in parts.items()}
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
