import numpy as np


class Normalizer:
    def __init__(self, min_vals, max_vals):
        self.min = np.array(min_vals)
        self.max = np.array(max_vals)
        # prevent division by zero
        self.max[self.max == self.min] += 1e-8

    @classmethod
    def from_data(cls, data):
        min_vals = np.min(data, axis=(0, 1))
        max_vals = np.max(data, axis=(0, 1))
        return cls(min_vals, max_vals)

    @classmethod
    def from_stats(cls, stats, key_prefix):
        return cls(stats[f"{key_prefix}_min"], stats[f"{key_prefix}_max"])

    def normalize(self, x):
        norm = (x - self.min) / (self.max - self.min)
        # scale to [-1, 1]
        return 2 * norm - 1

    def denormalize(self, x):
        norm = (x + 1) / 2
        return norm * (self.max - self.min) + self.min


def save_normalization(path, obs_norm, act_norm):
    np.savez(
        path,
        obs_min=obs_norm.min,
        obs_max=obs_norm.max,
        action_min=act_norm.min,
        action_max=act_norm.max,
    )


def load_normalization(path):
    stats = np.load(path)
    obs_norm = Normalizer.from_stats(stats, "obs")
    act_norm = Normalizer.from_stats(stats, "action")
    return obs_norm, act_norm
