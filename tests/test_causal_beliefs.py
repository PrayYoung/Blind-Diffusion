import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from blind_diffusion.data.robomimic_dataset import RoboMimicSequenceDataset
from blind_diffusion.train.beliefs import compute_pre_action_beliefs, previous_actions
from blind_diffusion.models.rssm import RSSM


def _make_lowdim_hdf5(path: Path):
    with h5py.File(path, "w") as h5:
        data = h5.create_group("data")
        for demo_idx in range(2):
            demo = data.create_group(f"demo_{demo_idx}")
            obs = demo.create_group("obs")
            base = np.arange(5, dtype=np.float32)[:, None] + demo_idx * 100
            obs.create_dataset("state", data=np.concatenate([base, base + 1, base + 2], axis=1))
            demo.create_dataset("actions", data=np.concatenate([base, base + 10], axis=1))
            demo.create_dataset("rewards", data=np.arange(5, dtype=np.float32))
            demo.create_dataset("dones", data=np.zeros((5,), dtype=np.float32))


def _world_model():
    model = SimpleNamespace()
    model.encoder = torch.nn.Linear(4, 8)
    model.rssm = RSSM(action_dim=2, obs_dim=8, deter_dim=7, stoch_dim=3, hidden_dim=11)
    return model


class CausalBeliefTests(unittest.TestCase):
    def test_previous_actions_aligns_decision_state_with_current_observation(self):
        actions = torch.tensor([[[1.0], [2.0], [3.0]]])
        self.assertTrue(torch.equal(previous_actions(actions), torch.tensor([[[0.0], [1.0], [2.0]]])))

    def test_pre_action_belief_does_not_leak_chunk_first_action(self):
        wm = _world_model()
        obs = torch.randn(1, 5, 4)
        actions = torch.randn(1, 5, 2)
        changed = actions.clone()
        changed[:, 2] += 100.0

        torch.manual_seed(123)
        original_beliefs = compute_pre_action_beliefs(wm, obs, actions)
        torch.manual_seed(123)
        changed_beliefs = compute_pre_action_beliefs(wm, obs, changed)

        # Belief at t=2 only uses obs[:=2] and actions[:2], never actions[2].
        self.assertTrue(torch.allclose(original_beliefs[:, 2], changed_beliefs[:, 2]))
        # The changed action becomes the previous action at the following step.
        self.assertFalse(torch.allclose(original_beliefs[:, 3], changed_beliefs[:, 3]))

    def test_dataset_keeps_native_actions_and_normalizes_observations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lowdim.hdf5"
            _make_lowdim_hdf5(path)
            dataset = RoboMimicSequenceDataset(
                str(path), ["state"], seq_len=3, normalize_obs=True, normalize_action=False
            )
            sample = dataset[0]

            self.assertIsNotNone(dataset.obs_mean)
            self.assertIsNotNone(dataset.obs_std)
            self.assertIsNone(dataset.act_mean)
            self.assertIsNone(dataset.act_std)
            self.assertTrue(torch.equal(sample["actions"], torch.tensor([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]])))


if __name__ == "__main__":
    unittest.main()
