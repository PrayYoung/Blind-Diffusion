"""Common, resumable simulator protocol for the matched baseline campaign."""
import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from filelock import FileLock
from omegaconf import OmegaConf
import torch

from blind_diffusion.baselines.open_loop_bc import BCPolicy
from blind_diffusion.diffusion.policy import build_action_policy
from blind_diffusion.eval.protocol import is_blackout, task_success
from blind_diffusion.train.train_world_model import WorldModel
from blind_diffusion.utils.checkpoint import load_checkpoint
from blind_diffusion.utils.seed import set_seed


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=16)
def file_digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def vector(obs, keys):
    return np.concatenate([np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in keys])


class MatchedAgent:
    def __init__(self, method, world_model, checkpoint, device="cpu"):
        self.method, self.device = method, torch.device(device)
        wm = load_checkpoint(world_model, map_location=self.device)
        self.config = OmegaConf.create(wm["config"])
        self.keys = list(self.config.task.obs_keys)
        self.obs_dim = len(wm["norm"]["obs_mean"])
        self.action_dim = wm["model"]["rssm.gru.weight_ih"].shape[1] - self.config.model.rssm.stoch_dim
        self.mean, self.std = [torch.as_tensor(wm["norm"][key], device=self.device, dtype=torch.float32)
                               for key in ("obs_mean", "obs_std")]
        self.wm = WorldModel(self.obs_dim, self.action_dim, self.config.model).to(self.device).eval()
        self.wm.load_state_dict(wm["model"])
        self.horizon = 8
        self.policy = None
        self.provenance = {"world_model_sha256": file_digest(world_model),
                           "evaluation_engine_sha256": file_digest(__file__)}
        if method != "mpc":
            payload = load_checkpoint(checkpoint, map_location=self.device)
            if payload["split"] != wm["split"]:
                raise ValueError("Policy and world model must share their complete episode split")
            self.provenance["policy_sha256"] = file_digest(checkpoint)
            if method == "bc":
                if payload.get("kind") != "reactive_bc":
                    raise ValueError("Require the matched held-out reactive BC checkpoint")
                for key in ("obs_mean", "obs_std"):
                    if not np.array_equal(payload["norm"][key], wm["norm"][key]):
                        raise ValueError("BC and generative models must share normalization")
                self.policy = BCPolicy(self.obs_dim, self.action_dim, payload["hidden_dim"]).to(self.device)
            else:
                expected = "flow_matching" if method.startswith("cfm") else "diffusion"
                if payload["config"].get("policy_type", "diffusion") != expected:
                    raise ValueError("Policy type does not match the requested method")
                self.policy = build_action_policy(payload["config"], self.action_dim,
                    self.config.model.rssm.deter_dim + self.config.model.rssm.stoch_dim).to(self.device)
            self.policy.load_state_dict(payload.get("ema_model", payload["model"]))
            self.policy.eval()
        self.reset()

    def reset(self):
        self.state = {"h": torch.zeros(1, self.config.model.rssm.deter_dim, device=self.device),
                      "z": torch.zeros(1, self.config.model.rssm.stoch_dim, device=self.device)}
        self.previous_action = torch.zeros(1, self.action_dim, device=self.device)
        self.plan, self.plan_index = None, 0

    def _generator(self, seed, step):
        return torch.Generator(device=self.device).manual_seed(1000003 * seed + step + 7919)

    @torch.no_grad()
    def cem(self, generator):
        horizon, population, elites, iterations = 15, 256, 32, 4
        mean = torch.zeros(horizon, self.action_dim, device=self.device)
        std = torch.ones_like(mean)
        start = {key: value.repeat(population, 1) for key, value in self.state.items() if key in ("h", "z")}
        for _ in range(iterations):
            actions = (mean + std * torch.randn((population, horizon, self.action_dim),
                       device=self.device, generator=generator)).clamp(-1, 1)
            imagined = self.wm.rssm.imagine(actions, start, deterministic=True)
            score = self.wm.reward_head(torch.cat([imagined["h"], imagined["z"]], -1)).sum(-1)
            selected = actions[score.topk(elites).indices]
            mean = .1 * selected.mean(0) + .9 * mean
            std = .1 * (selected.std(0) + 1e-6) + .9 * std
        return mean[0:1]

    @torch.no_grad()
    def action(self, observation, blind, seed, step):
        if self.method == "bc":
            inputs = (torch.zeros((1, self.obs_dim), device=self.device) if blind else
                      (torch.as_tensor(observation, device=self.device)[None] - self.mean) / self.std)
            return self.policy(inputs).clamp(-1, 1)
        if blind and self.method != "open_loop":
            # Do not even encode the true observation on this path.
            state = self.wm.rssm.imagine_step(self.previous_action, self.state, deterministic=True)
        else:
            inputs = (torch.zeros((1, self.obs_dim), device=self.device) if blind else
                      (torch.as_tensor(observation, device=self.device)[None] - self.mean) / self.std)
            state = self.wm.rssm.observe_step(self.wm.encoder(inputs), self.previous_action,
                                             self.state, deterministic=True)
        self.state = {key: state[key] for key in ("h", "z")}
        if self.method == "mpc":
            action = self.cem(self._generator(seed, step))
        elif self.method == "open_loop" and self.plan is not None and self.plan_index < self.horizon:
            action = self.plan[:, :, self.plan_index]
            self.plan_index += 1
        else:
            belief = torch.cat([self.state["h"], self.state["z"]], -1)
            shape = (1, self.action_dim, self.horizon)
            noise = torch.randn(shape, device=self.device, generator=self._generator(seed, step))
            is_flow = self.method.startswith("cfm")
            steps = (2 if self.method == "cfm2" else 4) if is_flow else 10
            sequence = self.policy.sample(shape, belief, "euler" if is_flow else "ddim", steps,
                                          initial_noise=noise)
            action = sequence[:, :, 0]
            if self.method == "open_loop":
                self.plan, self.plan_index = sequence, 1
        self.previous_action = action.clamp(-1, 1)
        return self.previous_action


@lru_cache(maxsize=8)
def worker_assets(dataset, method, wm, policy, device):
    """One environment/model per worker and method; seeds are reset per episode."""
    import robomimic.utils.obs_utils as ObsUtils
    from blind_diffusion.env.robomimic_env import make_env
    torch.set_num_threads(1)
    set_seed(0)
    agent = MatchedAgent(method, wm, policy, device)
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": agent.keys, "rgb": []}})
    env = make_env(dataset)
    if env.action_dimension != agent.action_dim:
        raise ValueError("Dataset controller and checkpoint action dimensions differ")
    return env, agent


def evaluate_episode(job):
    protocol = job["protocol"]
    protocol_hash = digest_json(protocol)
    output = Path(job["output"])
    expected = {key: job[key] for key in ("task", "method", "duration", "seed")}
    expected["protocol_sha256"] = protocol_hash
    if output.exists():
        saved = json.loads(output.read_text())
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing episode has incompatible benchmark identity")
        if saved["provenance"]["world_model_sha256"] != file_digest(job["wm"]):
            raise ValueError("World model changed since this episode was completed")
        if job.get("policy") and saved["provenance"]["policy_sha256"] != file_digest(job["policy"]):
            raise ValueError("Policy changed since this episode was completed")
        return saved
    env, agent = worker_assets(job["dataset"], job["method"], job["wm"], job.get("policy"), job["device"])
    set_seed(job["seed"])
    agent.reset()
    obs = env.reset()
    if job.get("initial_states_dir"):
        template_path = Path(job["initial_states_dir"])/job["task"]/f"seed{job['seed']}.json"
        template_path.parent.mkdir(parents=True,exist_ok=True)
        with FileLock(str(template_path)+".lock",timeout=60):
            if template_path.exists():
                template = json.loads(template_path.read_text())
            else:
                fresh = env.get_state()
                template = {"states":np.asarray(fresh["states"],dtype="<f8").tolist(),
                            "model":fresh.get("model"),"seed":job["seed"],
                            "dataset_sha256":file_digest(job["dataset"])}
                temporary = template_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(template,allow_nan=False)+"\n")
                os.replace(temporary,template_path)
            if template["seed"] != job["seed"] or template["dataset_sha256"] != file_digest(job["dataset"]):
                raise ValueError("Canonical initial state belongs to another seed or dataset")
            restored = {"states":np.asarray(template["states"],dtype=np.float64)}
            if template.get("model") is not None:
                restored["model"] = template["model"]
            # Apply the same reset-to-state procedure even to the first method.
            obs = env.reset_to(restored)
            actual_state=np.asarray(env.get_state()["states"],dtype="<f8")
            actual_observation=vector(obs,agent.keys)
            if "restored_states" not in template:
                # MuJoCo forward() can renormalize quaternion coordinates by
                # one ULP. Always restore the original input, and compare its
                # actual output across methods, rather than requiring an
                # idempotent raw state. No tolerance is used for pairing.
                template["restored_states"]=actual_state.tolist()
                template["restored_observation"]=actual_observation.tolist()
                temporary=template_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(template,allow_nan=False)+"\n")
                os.replace(temporary,template_path)
            if not np.array_equal(actual_state,np.asarray(template["restored_states"],dtype=np.float64)):
                raise ValueError("Simulator restored a different canonical starting state across conditions")
            if not np.array_equal(actual_observation,np.asarray(template["restored_observation"],dtype=np.float32)):
                raise ValueError("Simulator restored different canonical observations across conditions")
    initial = vector(obs, agent.keys).tolist()
    simulation_state = np.asarray(env.get_state()["states"], dtype="<f8")
    state_hash = hashlib.sha256(simulation_state.tobytes()).hexdigest()
    total, success, timings, steps = 0., False, [], 0
    termination = "step_limit"
    limit = protocol["max_steps"][job["task"]]
    started = time.monotonic()
    for step in range(limit):
        blind = is_blackout(step, protocol["visible_warmup"], protocol["blackout_period"], job["duration"])
        tick = time.perf_counter()
        action = agent.action(vector(obs, agent.keys), blind, job["seed"], step)
        native_action = action.squeeze(0).cpu().numpy()
        timings.append(time.perf_counter() - tick)
        if not np.isfinite(native_action).all():
            # A numerically unstable controller is a scientific failure,
            # not an infrastructure exception to rerun until it succeeds.
            termination = "nonfinite_action"
            break
        obs, reward, done, info = env.step(native_action)
        steps += 1
        total += float(reward)
        success = task_success(env, info)
        if success or done:
            termination = "success" if success else "environment_done"
            break
    result = {**expected, "provenance": agent.provenance, "dataset_sha256": file_digest(job["dataset"]),
              "initial_observation": initial, "initial_simulator_state_sha256": state_hash,
              "success": bool(success), "return": total, "length": steps,
              "termination_reason": termination,
              "wall_seconds": time.monotonic() - started, "inference_seconds_mean": float(np.mean(timings)),
              "device": job["device"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, allow_nan=False) + "\n")
    os.replace(temporary, output)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, help="JSON episode job, including its frozen protocol")
    args = parser.parse_args()
    print(json.dumps(evaluate_episode(json.loads(Path(args.job).read_text())), allow_nan=False))


if __name__ == "__main__":
    main()
