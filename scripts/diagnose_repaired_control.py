#!/usr/bin/env python3
"""Trace-only diagnostic for an existing low-dimensional MPC checkpoint.

This intentionally reuses the production MPC evaluation path while emitting
per-step control, environment, reward, and latent-state evidence. It neither
trains nor alters the world model or planner.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import robomimic.utils.obs_utils as ObsUtils
from omegaconf import OmegaConf

from blind_diffusion.env.robomimic_env import make_env
from blind_diffusion.eval.evaluate_lowdim import _denormalize, _normalize, _obs_to_vec
from blind_diffusion.planner.mpc import build_planner
from blind_diffusion.train.train_world_model import WorldModel
from blind_diffusion.utils.checkpoint import load_checkpoint


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_list(value):
    return np.asarray(value).reshape(-1).astype(float).tolist()


def _state_summary(state, previous):
    h = state["h"].detach()
    z = state["z"].detach()
    out = {
        "h_norm": float(torch.linalg.vector_norm(h).item()),
        "z_norm": float(torch.linalg.vector_norm(z).item()),
    }
    if previous is None:
        out.update({"h_delta_norm": None, "z_delta_norm": None})
    else:
        out.update({
            "h_delta_norm": float(torch.linalg.vector_norm(h - previous["h"]).item()),
            "z_delta_norm": float(torch.linalg.vector_norm(z - previous["z"]).item()),
        })
    return out


def _planned_objective(planner, state, wm, terminal_penalty, constraint_penalty, action):
    """Return a labelled proxy objective for repeating the chosen first action."""
    if not hasattr(planner, "_evaluate"):
        return None
    actions = action.detach().reshape(1, 1, -1).repeat(1, planner.horizon, 1)
    with torch.no_grad():
        value = planner._evaluate(actions, state, wm, terminal_penalty, constraint_penalty)
    return float(value.reshape(-1)[0].item())


def _rollout(env, wm, planner, cfg, norm, policy, episode_seed, max_steps, action_dim, device):
    _seed_everything(episode_seed)
    if hasattr(env, "seed"):
        env.seed(episode_seed)
    obs = env.reset()
    state = {"h": torch.zeros(1, cfg.model.rssm.deter_dim, device=device),
             "z": torch.zeros(1, cfg.model.rssm.stoch_dim, device=device)}
    prev_action = torch.zeros(1, action_dim, device=device)
    action_low = torch.full((action_dim,), -1.0, device=device)
    action_high = torch.full((action_dim,), 1.0, device=device)
    trace, total_return, success_any = [], 0.0, False
    previous_state = None

    for step in range(max_steps):
        obs_vec = _obs_to_vec(obs, cfg.task.obs_keys).unsqueeze(0).to(device)
        obs_embed = wm.encoder(_normalize(obs_vec, norm.get("obs_mean"), norm.get("obs_std")))
        state = wm.rssm.observe_step(obs_embed, prev_action, state)
        latent = _state_summary(state, previous_state)

        if policy == "mpc":
            action_n = planner.plan(state, wm, cfg.planner.terminal_penalty, cfg.planner.constraint_penalty)
            objective = _planned_objective(planner, state, wm, cfg.planner.terminal_penalty,
                                           cfg.planner.constraint_penalty, action_n)
        elif policy == "zero":
            action_n, objective = torch.zeros(action_dim, device=device), None
        elif policy == "random":
            action_n, objective = torch.empty(action_dim, device=device).uniform_(-1.0, 1.0), None
        else:
            raise ValueError(f"Unsupported policy: {policy}")

        action_env = torch.clamp(_denormalize(action_n, norm.get("act_mean"), norm.get("act_std")),
                                 action_low, action_high)
        # Environment wrappers may reuse observation buffers across step calls.
        # Copy before stepping so this trace measures physical response rather
        # than an alias-mutated before/after pair.
        eef = np.asarray(obs["robot0_eef_pos"], dtype=float).copy()
        next_obs, reward, done, info = env.step(action_env.detach().cpu().numpy())
        next_eef = np.asarray(next_obs["robot0_eef_pos"], dtype=float).copy()
        state_change = float(np.linalg.norm(next_eef - eef))
        step_success = bool(info.get("success", False))
        trace.append({
            "step": step,
            "action_normalized": _as_list(action_n.detach().cpu()),
            "action_environment": _as_list(action_env.detach().cpu()),
            "eef_position": _as_list(eef),
            "next_eef_position": _as_list(next_eef),
            "eef_delta_norm": state_change,
            "reward": float(reward),
            "success": step_success,
            "done": bool(done),
            "terminated": bool(done),
            "truncated": bool((step + 1 == max_steps) and not done),
            "time_limit_reached": bool(step + 1 == max_steps),
            "planned_objective_repeat_action_proxy": objective,
            **latent,
        })
        total_return += float(reward)
        success_any = success_any or step_success
        previous_state = {key: value.detach().clone() for key, value in state.items()}
        prev_action, obs = action_n.detach().reshape(1, -1), next_obs
        if done:
            break

    actions = np.asarray([x["action_environment"] for x in trace], dtype=float)
    eef_delta = np.asarray([x["eef_delta_norm"] for x in trace], dtype=float)
    latent_delta = np.asarray([x["h_delta_norm"] for x in trace[1:]], dtype=float)
    return {
        "policy": policy,
        "episode_seed": episode_seed,
        "steps": len(trace),
        "return": total_return,
        "success": success_any,
        "action_mean": _as_list(actions.mean(axis=0)),
        "action_std": _as_list(actions.std(axis=0)),
        "action_min": _as_list(actions.min(axis=0)),
        "action_max": _as_list(actions.max(axis=0)),
        "actions_vary_over_time": bool(len(actions) > 1 and np.any(np.ptp(actions, axis=0) > 1e-6)),
        "eef_delta_mean": float(eef_delta.mean()),
        "eef_delta_max": float(eef_delta.max()),
        "latent_h_delta_mean": float(latent_delta.mean()) if len(latent_delta) else 0.0,
        "latent_h_delta_max": float(latent_delta.max()) if len(latent_delta) else 0.0,
        "trace": trace,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--policies", nargs="+", choices=("mpc", "zero", "random"),
                        default=("mpc", "zero", "random"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    cfg = OmegaConf.create(checkpoint["config"])
    # World-model checkpoints are written by train_world_model and therefore
    # intentionally carry no planner settings. Match the production MPC
    # evaluator's default (eval_mpc -> planner/cem) without changing either.
    planner_cfg = OmegaConf.load(
        Path(__file__).resolve().parents[1] / "src/blind_diffusion/configs/planner/cem.yaml"
    )
    cfg = OmegaConf.merge(cfg, {"planner": planner_cfg})
    task = cfg.task
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": list(task.obs_keys), "rgb": []}})
    hdf5_path = os.path.join(os.environ["ROBO_DATA"], task.hdf5_name)
    env = make_env(hdf5_path)
    initial_obs = env.reset()
    obs_dim = sum(initial_obs[key].shape[0] for key in task.obs_keys)
    action_dim = env.action_dimension
    wm = WorldModel(obs_dim, action_dim, cfg.model).to(device)
    wm.load_state_dict(checkpoint["model"])
    wm.eval()
    planner = build_planner(cfg, action_dim)
    norm = checkpoint.get("norm", {})

    # The episode seed is fixed within a checkpoint trace.
    episode_seed = 17000 + args.checkpoint_seed
    with torch.no_grad():
        rollouts = [_rollout(env, wm, planner, cfg, norm, policy, episode_seed, args.max_steps,
                             action_dim, device) for policy in args.policies]
    payload = {"checkpoint": args.checkpoint, "checkpoint_seed": args.checkpoint_seed,
               "max_steps": args.max_steps, "rollouts": rollouts}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
