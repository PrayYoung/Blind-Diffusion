import torch
import torch.nn as nn
from .modules import mlp


class RewardHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = mlp(in_dim, 1, hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DoneHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = mlp(in_dim, 1, hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ObsHead(nn.Module):
    def __init__(self, in_dim: int, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.net = mlp(in_dim, obs_dim, hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
