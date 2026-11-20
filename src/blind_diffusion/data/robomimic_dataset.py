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
        pad_end: int = 0,
        demo_names=None,
        norm_stats=None,
    ):
        assert os.path.exists(hdf5_path), f"Missing dataset: {hdf5_path}"
        self.hdf5_path = hdf5_path
        self.obs_keys = obs_keys
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.normalize_obs = normalize_obs
        self.normalize_action = normalize_action
        self.pad_end = pad_end

        self._h5 = None
        with h5py.File(self.hdf5_path, "r") as h5:
            demo_names = sorted(h5["data"].keys()) if demo_names is None else list(demo_names)
            if max_demos is not None:
                demo_names = demo_names[:max_demos]
            self.demo_names = demo_names
            self.demo_lens = [h5[f"data/{d}/actions"].shape[0] for d in demo_names]
            self.demo_len_by_name = dict(zip(self.demo_names, self.demo_lens))
            self.index = self._build_index()
            obs_all, act_all = self._gather_stats_samples(h5)
        self.obs_mean, self.obs_std = compute_mean_std(obs_all) if normalize_obs else (None, None)
        self.act_mean, self.act_std = compute_mean_std(act_all) if normalize_action else (None, None)
        if norm_stats is not None:
            self.obs_mean, self.obs_std = norm_stats["obs_mean"], norm_stats["obs_std"]
            self.act_mean, self.act_std = norm_stats["act_mean"], norm_stats["act_std"]

    def _build_index(self) -> List[Tuple[str, int]]:
        idx = []
        need = self.seq_len
        for d, length in zip(self.demo_names, self.demo_lens):
            max_start = min(length - 1, length - need + self.pad_end)
            for s in range(max_start + 1):
                idx.append((d, s))
        return idx

    def _gather_stats_samples(self, h5: h5py.File) -> Tuple[np.ndarray, np.ndarray]:
        obs_list = []
        act_list = []
        for d in self.demo_names:
            obs = self._load_obs(h5, d)
            act = h5[f"data/{d}/actions"][:]
            obs_list.append(obs)
            act_list.append(act)
        obs_all = np.concatenate(obs_list, axis=0)
        act_all = np.concatenate(act_list, axis=0)
        return obs_all, act_all

    def _load_obs(self, h5: h5py.File, demo_name: str) -> np.ndarray:
        obs_parts = []
        for k in self.obs_keys:
            obs_parts.append(h5[f"data/{demo_name}/obs/{k}"][:])
        return np.concatenate(obs_parts, axis=-1)

    def _ensure_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def _close_h5(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self):
        self._close_h5()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        h5 = self._ensure_h5()
        demo, start = self.index[idx]
        end = start + self.seq_len
        demo_len = self.demo_len_by_name[demo]
        valid = np.arange(self.seq_len) + start < demo_len
        obs = self._load_obs(h5, demo)[start:min(end, demo_len)]
        actions = h5[f"data/{demo}/actions"][start:min(end, demo_len)]
        rewards = h5[f"data/{demo}/rewards"][start:min(end, demo_len)]
        dones = h5[f"data/{demo}/dones"][start:min(end, demo_len)]
        if len(obs) < self.seq_len:
            pad = self.seq_len - len(obs)
            obs = np.concatenate([obs, np.repeat(obs[-1:], pad, axis=0)], axis=0)
            actions = np.concatenate([actions, np.zeros((pad, actions.shape[-1]), dtype=actions.dtype)], axis=0)
            rewards = np.concatenate([rewards, np.zeros(pad, dtype=rewards.dtype)], axis=0)
            dones = np.concatenate([dones, np.ones(pad, dtype=dones.dtype)], axis=0)

        if self.normalize_obs:
            obs = normalize(obs, self.obs_mean, self.obs_std)
        if self.normalize_action:
            actions = normalize(actions, self.act_mean, self.act_std)

        return {
            "obs": torch.from_numpy(obs).float(),
            "actions": torch.from_numpy(actions).float(),
            "rewards": torch.from_numpy(rewards).float(),
            "dones": torch.from_numpy(dones).float(),
            "valid": torch.from_numpy(valid),
            "is_start": torch.tensor(start == 0),
        }

    def get_norm_stats(self):
        return {
            "obs_mean": self.obs_mean,
            "obs_std": self.obs_std,
            "act_mean": self.act_mean,
            "act_std": self.act_std,
        }


def build_episode_datasets(hdf5_path, obs_keys, seq_len, burn_in=0,
                           normalize_obs=True, normalize_action=False,
                           pad_end=0, val_fraction=0.05, seed=0, norm_stats=None,
                           split=None):
    """Shared split for both stages; fit normalization on training episodes only."""
    with h5py.File(hdf5_path, "r") as handle:
        names = sorted(handle["data"].keys())
    if split is None:
        if len(names) < 2 or not 0 < val_fraction < 1:
            raise ValueError("Episode validation requires >=2 demos and 0<val_fraction<1")
        order = np.random.default_rng(seed).permutation(len(names))
        n_val = min(len(names) - 1, max(1, round(len(names) * val_fraction)))
        val_names = [names[i] for i in order[:n_val]]
        train_names = [names[i] for i in order[n_val:]]
        split = {"train": train_names, "val": val_names}
    if (set(split["train"]) & set(split["val"]) or
            set(split["train"]) | set(split["val"]) != set(names)):
        raise ValueError("Split must partition all dataset demos without overlap")
    kwargs = dict(hdf5_path=hdf5_path, obs_keys=obs_keys, seq_len=seq_len,
                  burn_in=burn_in, normalize_obs=normalize_obs,
                  normalize_action=normalize_action, pad_end=pad_end)
    train = RoboMimicSequenceDataset(**kwargs, demo_names=split["train"], norm_stats=norm_stats)
    val = RoboMimicSequenceDataset(**kwargs, demo_names=split["val"], norm_stats=train.get_norm_stats())
    if not len(train) or not len(val):
        raise ValueError("Training and validation both require eligible windows")
    return train, val, split
