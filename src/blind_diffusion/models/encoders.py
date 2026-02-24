import torch
import torch.nn as nn
from .modules import mlp


class MLPEncoder(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, layers: int):
        super().__init__()
        self.net = mlp(obs_dim, hidden_dim, hidden_dim, layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
