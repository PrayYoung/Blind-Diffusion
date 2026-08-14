import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from blind_diffusion.data.robomimic_dataset import RoboMimicSequenceDataset
from blind_diffusion.data.robomimic_dataset_image import RoboMimicImageSequenceDataset

torch.multiprocessing.set_sharing_strategy("file_system")


def _make_lowdim_hdf5(path: Path, demos: int = 2, steps: int = 5):
    with h5py.File(path, "w") as h5:
        data = h5.create_group("data")
        for demo_idx in range(demos):
            demo = data.create_group(f"demo_{demo_idx}")
            obs = demo.create_group("obs")
            base = np.arange(steps, dtype=np.float32)[:, None] + demo_idx * 100
            obs.create_dataset("state", data=np.concatenate([base, base + 1, base + 2], axis=1))
            demo.create_dataset("actions", data=np.concatenate([base, base + 10], axis=1))
            demo.create_dataset("rewards", data=np.arange(steps, dtype=np.float32))
            demo.create_dataset("dones", data=np.zeros((steps,), dtype=np.float32))


def _make_image_hdf5(path: Path, demos: int = 2, steps: int = 5):
    with h5py.File(path, "w") as h5:
        data = h5.create_group("data")
        for demo_idx in range(demos):
            demo = data.create_group(f"demo_{demo_idx}")
            obs = demo.create_group("obs")
            images = np.full((steps, 8, 8, 3), fill_value=demo_idx * 10, dtype=np.uint8)
            lowdim = np.arange(steps, dtype=np.float32)[:, None] + demo_idx * 100
            obs.create_dataset("agentview_image", data=images)
            obs.create_dataset("state", data=np.concatenate([lowdim, lowdim + 1, lowdim + 2], axis=1))
            demo.create_dataset("actions", data=np.concatenate([lowdim, lowdim + 10], axis=1))
            demo.create_dataset("rewards", data=np.arange(steps, dtype=np.float32))
            demo.create_dataset("dones", data=np.zeros((steps,), dtype=np.float32))


def _numpy_collate(batch):
    return {
        key: np.stack([item[key].numpy() for item in batch], axis=0)
        for key in batch[0]
    }


class DatasetTests(unittest.TestCase):
    def test_lowdim_dataset_initializes_and_reads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lowdim.hdf5"
            _make_lowdim_hdf5(path)
            dataset = RoboMimicSequenceDataset(
                str(path),
                ["state"],
                seq_len=3,
                normalize_obs=False,
                normalize_action=False,
            )

            self.assertEqual(len(dataset), 6)
            sample = dataset[0]
            self.assertEqual(tuple(sample["obs"].shape), (3, 3))
            self.assertEqual(tuple(sample["actions"].shape), (3, 2))

    def test_lowdim_dataset_multiworker_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lowdim.hdf5"
            _make_lowdim_hdf5(path)
            dataset = RoboMimicSequenceDataset(
                str(path),
                ["state"],
                seq_len=3,
                normalize_obs=False,
                normalize_action=False,
            )

            for workers in (0, 1, 2, 4):
                loader = DataLoader(
                    dataset,
                    batch_size=2,
                    shuffle=False,
                    num_workers=workers,
                    collate_fn=_numpy_collate,
                )
                first_values = []
                for _ in range(2):
                    epoch_values = []
                    for batch in loader:
                        epoch_values.extend(batch["obs"][:, 0, 0].tolist())
                    first_values.append(epoch_values)
                self.assertEqual(first_values[0], first_values[1])
                self.assertEqual(len(first_values[0]), len(dataset))

    def test_image_dataset_initializes_and_reads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "image.hdf5"
            _make_image_hdf5(path)
            dataset = RoboMimicImageSequenceDataset(
                str(path),
                ["agentview_image"],
                ["state"],
                seq_len=3,
                normalize_action=False,
                normalize_lowdim=False,
                augment=False,
            )

            self.assertEqual(len(dataset), 6)
            sample = dataset[0]
            self.assertEqual(tuple(sample["images"].shape), (3, 3, 8, 8))
            self.assertEqual(tuple(sample["lowdim"].shape), (3, 3))
            self.assertEqual(tuple(sample["actions"].shape), (3, 2))


if __name__ == "__main__":
    unittest.main()
