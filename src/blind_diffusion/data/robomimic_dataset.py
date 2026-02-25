import os
from typing import List, Dict, Tuple
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import compute_mean_std, normalize


class RoboMimicSequenceDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        obs_keys: List[str],
        seq_len: int,
        burn_in: int = 0,
        normalize_obs: bool = True,
        normalize_action: bool = True,
        max_demos: int = None,
    ):
        assert os.path.exists(hdf5_path), f"Missing dataset: {hdf5_path}"
        self.hdf5_path = hdf5_path
        self.obs_keys = obs_keys
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.normalize_obs = normalize_obs
        self.normalize_action = normalize_action

        self._h5 = h5py.File(hdf5_path, "r")
        demo_names = sorted(list(self._h5["data"].keys()))
        if max_demos is not None:
            demo_names = demo_names[:max_demos]
        self.demo_names = demo_names

        self.demo_lens = [self._h5[f"data/{d}/actions"].shape[0] for d in demo_names]
        self.index = self._build_index()

        obs_all, act_all = self._gather_stats_samples()
        self.obs_mean, self.obs_std = compute_mean_std(obs_all) if normalize_obs else (None, None)
        self.act_mean, self.act_std = compute_mean_std(act_all) if normalize_action else (None, None)

    def _build_index(self) -> List[Tuple[str, int]]:
        idx = []
        need = self.seq_len
        for d, length in zip(self.demo_names, self.demo_lens):
            max_start = length - need
            for s in range(max_start + 1):
                idx.append((d, s))
        return idx

    def _gather_stats_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        obs_list = []
        act_list = []
        for d in self.demo_names:
            obs = self._load_obs(d)
            act = self._h5[f"data/{d}/actions"][:]
            obs_list.append(obs)
            act_list.append(act)
        obs_all = np.concatenate(obs_list, axis=0)
        act_all = np.concatenate(act_list, axis=0)
        return obs_all, act_all

    def _load_obs(self, demo_name: str) -> np.ndarray:
        obs_parts = []
        for k in self.obs_keys:
            obs_parts.append(self._h5[f"data/{demo_name}/obs/{k}"][:])
        return np.concatenate(obs_parts, axis=-1)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        demo, start = self.index[idx]
        end = start + self.seq_len
        obs = self._load_obs(demo)[start:end]
        actions = self._h5[f"data/{demo}/actions"][start:end]
        rewards = self._h5[f"data/{demo}/rewards"][start:end]
        dones = self._h5[f"data/{demo}/dones"][start:end]

        if self.normalize_obs:
            obs = normalize(obs, self.obs_mean, self.obs_std)
        if self.normalize_action:
            actions = normalize(actions, self.act_mean, self.act_std)

        return {
            "obs": torch.from_numpy(obs).float(),
            "actions": torch.from_numpy(actions).float(),
            "rewards": torch.from_numpy(rewards).float(),
            "dones": torch.from_numpy(dones).float(),
        }

    def get_norm_stats(self):
        return {
            "obs_mean": self.obs_mean,
            "obs_std": self.obs_std,
            "act_mean": self.act_mean,
            "act_std": self.act_std,
        }
