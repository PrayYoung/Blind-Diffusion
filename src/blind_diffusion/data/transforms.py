import numpy as np


def compute_mean_std(x: np.ndarray, eps: float = 1e-6):
    mean = x.mean(axis=0)
    std = x.std(axis=0) + eps
    return mean, std


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (x - mean) / std
