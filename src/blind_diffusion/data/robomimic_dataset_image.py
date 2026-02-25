import os
from typing import List, Dict, Tuple, Optional
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import compute_mean_std, normalize


class RoboMimicImageSequenceDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        image_keys: List[str],
        lowdim_keys: Optional[List[str]],
        seq_len: int,
        burn_in: int = 0,
        normalize_action: bool = True,
        normalize_lowdim: bool = True,
        augment: bool = False,
        crop_size: Optional[int] = None,
        max_demos: int = None,
    ):
        assert os.path.exists(hdf5_path), f"Missing dataset: {hdf5_path}"
        self.hdf5_path = hdf5_path
        self.image_keys = image_keys
        self.lowdim_keys = lowdim_keys or []
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.normalize_action = normalize_action
        self.normalize_lowdim = normalize_lowdim
        self.augment = augment
        self.crop_size = crop_size

        self._h5 = h5py.File(hdf5_path, "r")
        demo_names = sorted(list(self._h5["data"].keys()))
        if max_demos is not None:
            demo_names = demo_names[:max_demos]
        self.demo_names = demo_names

        self.demo_lens = [self._h5[f"data/{d}/actions"].shape[0] for d in demo_names]
        self.index = self._build_index()

        low_all, act_all = self._gather_stats_samples()
        self.low_mean, self.low_std = compute_mean_std(low_all) if (self.lowdim_keys and normalize_lowdim) else (None, None)
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
        low_list = []
        act_list = []
        for d in self.demo_names:
            if self.lowdim_keys:
                low = self._load_lowdim(d)
                low_list.append(low)
            act = self._h5[f"data/{d}/actions"][:]
            act_list.append(act)
        low_all = np.concatenate(low_list, axis=0) if low_list else np.zeros((1, 1), dtype=np.float32)
        act_all = np.concatenate(act_list, axis=0)
        return low_all, act_all

    def _load_lowdim(self, demo_name: str) -> np.ndarray:
        parts = []
        for k in self.lowdim_keys:
            parts.append(self._h5[f"data/{demo_name}/obs/{k}"][:])
        return np.concatenate(parts, axis=-1)

    def _load_images(self, demo_name: str) -> np.ndarray:
        # returns [T, H, W, C] (single key) or concatenated along channel if multiple keys
        imgs = []
        for k in self.image_keys:
            imgs.append(self._h5[f"data/{demo_name}/obs/{k}"][:])
        if len(imgs) == 1:
            return imgs[0]
        # concat on channel dimension
        return np.concatenate(imgs, axis=-1)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        demo, start = self.index[idx]
        end = start + self.seq_len

        images = self._load_images(demo)[start:end]  # [T, H, W, C]
        actions = self._h5[f"data/{demo}/actions"][start:end]
        rewards = self._h5[f"data/{demo}/rewards"][start:end]
        dones = self._h5[f"data/{demo}/dones"][start:end]

        low = None
        if self.lowdim_keys:
            low = self._load_lowdim(demo)[start:end]
            if self.normalize_lowdim:
                low = normalize(low, self.low_mean, self.low_std)

        if self.normalize_action:
            actions = normalize(actions, self.act_mean, self.act_std)

        # images to float [0,1] and channel-first
        images = images.astype(np.float32) / 255.0
        images = np.transpose(images, (0, 3, 1, 2))

        images_t = torch.from_numpy(images).float()
        if self.augment:
            images_t = self._augment(images_t)

        out = {
            "images": images_t,
            "actions": torch.from_numpy(actions).float(),
            "rewards": torch.from_numpy(rewards).float(),
            "dones": torch.from_numpy(dones).float(),
        }
        if low is not None:
            out["lowdim"] = torch.from_numpy(low).float()
        return out

    def get_norm_stats(self):
        return {
            "low_mean": self.low_mean,
            "low_std": self.low_std,
            "act_mean": self.act_mean,
            "act_std": self.act_std,
        }

    def _augment(self, images: torch.Tensor) -> torch.Tensor:
        # images: [T, C, H, W]
        if self.crop_size is not None:
            _, _, h, w = images.shape
            if h >= self.crop_size and w >= self.crop_size:
                top = torch.randint(0, h - self.crop_size + 1, (1,)).item()
                left = torch.randint(0, w - self.crop_size + 1, (1,)).item()
                images = images[:, :, top : top + self.crop_size, left : left + self.crop_size]
        # random horizontal flip
        if torch.rand(()) < 0.5:
            images = torch.flip(images, dims=[3])
        return images
