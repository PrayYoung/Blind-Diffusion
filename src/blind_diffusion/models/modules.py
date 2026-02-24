import torch
import torch.nn as nn


def mlp(in_dim: int, out_dim: int, hidden_dim: int, layers: int, activation=nn.ELU):
    if layers <= 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))
    mods = [nn.Linear(in_dim, hidden_dim), activation()]
    for _ in range(layers - 2):
        mods += [nn.Linear(hidden_dim, hidden_dim), activation()]
    mods.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*mods)
