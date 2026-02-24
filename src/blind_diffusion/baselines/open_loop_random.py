import torch


class RandomPolicy:
    def __init__(self, action_dim, low=-1.0, high=1.0):
        self.action_dim = action_dim
        self.low = low
        self.high = high

    def __call__(self, obs):
        return (self.high - self.low) * torch.rand(self.action_dim) + self.low
