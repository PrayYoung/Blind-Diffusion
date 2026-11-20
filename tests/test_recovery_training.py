import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from blind_diffusion.data.robomimic_dataset import build_episode_datasets
from blind_diffusion.diffusion.diffusion import GaussianDiffusion
from blind_diffusion.diffusion.model import UNet1D
from blind_diffusion.eval.protocol import is_blackout, task_success
from blind_diffusion.models.rssm import RSSM
from blind_diffusion.train.beliefs import compute_pre_action_beliefs, sample_chunk_starts
from blind_diffusion.train.train_world_model import WorldModel, compute_loss
from blind_diffusion.utils.ema import ModelEMA


class RecoveryTrainingTests(unittest.TestCase):
    def test_two_stage_training_saves_validated_checkpoints(self):
        from blind_diffusion.train.train_world_model import main as train_wm
        from blind_diffusion.train.train_diffusion import main as train_policy
        from blind_diffusion.utils.checkpoint import load_checkpoint
        from unittest.mock import patch
        config_dir = str(Path(__file__).resolve().parents[1] / "src/blind_diffusion/configs")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.hdf5"
            with h5py.File(path, "w") as handle:
                for i in range(4):
                    group = handle.create_group(f"data/demo_{i}")
                    group.create_dataset("obs/state", data=np.full((10, 4), i * .1, dtype=np.float32))
                    group.create_dataset("actions", data=np.full((10, 2), .1, dtype=np.float32))
                    for key in ("rewards", "dones"):
                        group.create_dataset(key, data=np.zeros(10, dtype=np.float32))
            shared = [f"task.hdf5_name={path}", "task.obs_keys=[state]", "seq_len=8", "pad_end=2",
                "burn_in=1", "batch_size=4", "num_workers=0", f"run_dir={directory}",
                "model.encoder.hidden_dim=8", "model.rssm.deter_dim=7", "model.rssm.stoch_dim=3",
                "model.rssm.hidden_dim=11", "model.heads.reward_hidden=8", "model.heads.done_hidden=8", "model.heads.obs_hidden=8"]
            with initialize_config_dir(version_base=None, config_dir=config_dir):
                wm_cfg = compose(config_name="train_world_model_recovery", overrides=shared + [
                    "exp_name=wm", "max_steps=3", "val_every=2", "imag_horizon=3"])
                policy_cfg = compose(config_name="train_diffusion_recovery", overrides=shared + [
                    "exp_name=policy", f"wm_checkpoint={directory}/wm/checkpoints/best.pt", "epochs=1",
                    "diffusion.base_ch=8", "diffusion.horizon=4", "diffusion.timesteps=10", "val_mc_samples=1"])
                flow_cfg = compose(config_name="train_flow_matching_recovery", overrides=shared + [
                    "exp_name=flow", f"wm_checkpoint={directory}/wm/checkpoints/best.pt", "epochs=1", "max_train_steps=1",
                    "diffusion.base_ch=8", "diffusion.horizon=4", "val_mc_samples=1"])
            with patch("blind_diffusion.train.train_world_model.get_device", return_value=torch.device("cpu")), patch("blind_diffusion.train.train_diffusion.get_device", return_value=torch.device("cpu")):
                train_wm.__wrapped__(wm_cfg)
                train_policy.__wrapped__(policy_cfg)
                train_policy.__wrapped__(flow_cfg)
            logs = [json.loads(line) for line in (Path(directory) / "wm/logs.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in logs if "val_loss" in row], [2, 3])
            ckpt = load_checkpoint(f"{directory}/policy/checkpoints/best.pt")
            self.assertIn("ema_model", ckpt)
            self.assertFalse(set(ckpt["split"]["train"]) & set(ckpt["split"]["val"]))
            self.assertTrue(np.isfinite(ckpt["selection"]["value"]))
            flow_ckpt = load_checkpoint(f"{directory}/flow/checkpoints/best.pt")
            self.assertEqual(flow_ckpt["selection"]["metric"], "ema_heldout_velocity_loss")
            self.assertTrue(np.isfinite(flow_ckpt["selection"]["value"]))

    def test_split_and_statistics_exclude_validation_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.hdf5"
            with h5py.File(path, "w") as handle:
                for i, value in enumerate((0., 1000.)):
                    group = handle.create_group(f"data/demo_{i}")
                    group.create_dataset("obs/state", data=np.full((6, 2), value, dtype=np.float32))
                    group.create_dataset("actions", data=np.ones((6, 1), dtype=np.float32))
                    for key in ("rewards", "dones"):
                        group.create_dataset(key, data=np.zeros(6, dtype=np.float32))
            train, val, split = build_episode_datasets(str(path), ["state"], 4, pad_end=2)
            self.assertFalse(set(train.demo_names) & set(val.demo_names))
            expected = 0. if train.demo_names == ["demo_0"] else 1000.
            np.testing.assert_allclose(train.obs_mean, expected)
            np.testing.assert_array_equal(train.obs_mean, val.obs_mean)
            self.assertEqual(train[-1]["valid"].tolist(), [True, True, False, False])
            self.assertFalse(bool(train[-1]["is_start"]))
            self.assertTrue(bool(train[0]["is_start"]))

    def test_blackout_is_causal_and_ignores_hidden_observations(self):
        wm = SimpleNamespace(encoder=torch.nn.Linear(4, 5),
            rssm=RSSM(2, 5, 7, 3, 11))
        obs, actions = torch.randn(2, 8, 4), torch.randn(2, 8, 2)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[:, 3:6] = True
        first = compute_pre_action_beliefs(wm, obs, actions, True, mask)
        changed_obs, changed_actions = obs.clone(), actions.clone()
        changed_obs[:, 3:6] += 100
        changed_actions[:, 5:] += 100
        second = compute_pre_action_beliefs(wm, changed_obs, changed_actions, True, mask)
        self.assertTrue(torch.equal(first[:, :6], second[:, :6]))
        self.assertFalse(torch.equal(first[:, 6:], second[:, 6:]))

    def test_chunk_selection_respects_burnin_and_real_first_labels(self):
        valid = torch.tensor([[True] * 8 + [False] * 4] * 2)
        for _ in range(20):
            starts = sample_chunk_starts(valid, torch.tensor([True, False]), 4, 4)
            self.assertLess(int(starts[0]), 8)
            self.assertGreaterEqual(int(starts[1]), 4)
            self.assertLess(int(starts[1]), 8)

    def test_diffusion_loss_ignores_padded_labels(self):
        model = UNet1D(2, 8, 3, base_ch=8, architecture="skip_v2")
        diffusion = GaussianDiffusion(model, timesteps=10, prediction_type="x0")
        target, cond = torch.randn(2, 2, 8), torch.randn(2, 3)
        mask = torch.ones(2, 1, 8, dtype=torch.bool)
        mask[:, :, -2:] = False
        changed = target.clone()
        changed[:, :, -2:] = 100
        torch.manual_seed(4)
        first = diffusion.loss(target, cond, mask)
        torch.manual_seed(4)
        second = diffusion.loss(changed, cond, mask)
        self.assertTrue(torch.equal(first, second))
        first.backward()
        self.assertGreater(float(model.down1.conv1.weight.grad.abs().sum()), 0)

    def test_world_model_masks_padding_and_trains_recursive_priors(self):
        cfg = OmegaConf.create({"encoder": {"hidden_dim": 8, "layers": 2},
            "rssm": {"deter_dim": 7, "stoch_dim": 3, "hidden_dim": 11, "min_std": .1, "max_std": 1.},
            "heads": {"reward_hidden": 8, "done_hidden": 8, "obs_hidden": 8}})
        model = WorldModel(4, 2, cfg)
        batch = {"obs": torch.randn(2, 9, 4), "actions": torch.randn(2, 9, 2),
            "rewards": torch.zeros(2, 9), "dones": torch.zeros(2, 9),
            "valid": torch.tensor([[True] * 7 + [False] * 2] * 2)}
        torch.manual_seed(5)
        loss, metrics = compute_loss(batch, model, 1, .5, .1, imag_horizon=3, imag_weight=1., overshoot_weight=1.)
        changed = {k: v.clone() for k, v in batch.items()}
        for key in ("obs", "actions", "rewards", "dones"):
            changed[key][:, -2:] = 100
        torch.manual_seed(5)
        other, _ = compute_loss(changed, model, 1, .5, .1, imag_horizon=3, imag_weight=1., overshoot_weight=1.)
        self.assertTrue(torch.equal(loss, other))
        self.assertGreater(metrics["imag_obs_loss"], 0.)
        loss.backward()
        self.assertGreater(float(model.rssm.prior_net[0].weight.grad.abs().sum()), 0.)

    def test_sequence_determinism_and_legacy_checkpoint_loading(self):
        rssm = RSSM(2, 5, 7, 3, 11)
        obs, act = torch.randn(2, 8, 5), torch.randn(2, 8, 2)
        first = rssm.observe(obs, act, deterministic=True)
        second = rssm.observe(obs, act, deterministic=True)
        self.assertTrue(torch.equal(first["z"], second["z"]))
        state = {key: first[key][:, 0] for key in ("h", "z")}
        prior = rssm.imagine(act, state, deterministic=True)
        self.assertTrue(torch.equal(prior["z"], prior["prior_mean"]))
        legacy = UNet1D(2, 8, 3, 8)
        UNet1D(2, 8, 3, 8).load_state_dict(legacy.state_dict(), strict=True)

    def test_recovery_configs_and_success_protocol(self):
        config_dir = str(Path(__file__).resolve().parents[1] / "src/blind_diffusion/configs")
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            wm = compose(config_name="train_world_model_recovery")
            policy = compose(config_name="train_diffusion_recovery")
        self.assertEqual(wm.imag_horizon, 20)
        self.assertEqual(policy.diffusion.architecture, "skip_v2")
        self.assertFalse(is_blackout(4, 5, 20, 20))
        self.assertTrue(all(is_blackout(i, 5, 20, 20) for i in range(5, 50)))
        env = SimpleNamespace(is_success=lambda: {"task": True})
        self.assertTrue(task_success(env, {}))
        model = torch.nn.Linear(2, 2)
        ema = ModelEMA(model)
        before = ema.model.weight.clone()
        with torch.no_grad():
            model.weight.add_(1)
        ema.update(model)
        self.assertEqual(ema.updates, 1)
        self.assertTrue(torch.allclose(ema.model.weight, before + 9 / 11))


if __name__ == "__main__":
    unittest.main()
