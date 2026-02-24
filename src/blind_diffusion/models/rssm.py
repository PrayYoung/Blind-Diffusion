import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):
    def __init__(self, action_dim, obs_dim, deter_dim, stoch_dim, hidden_dim, min_std=0.1, max_std=1.0):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.hidden_dim = hidden_dim
        self.min_std = min_std
        self.max_std = max_std

        self.gru = nn.GRUCell(stoch_dim + action_dim, deter_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )

    def _get_stats(self, params: torch.Tensor):
        mean, std = torch.chunk(params, 2, dim=-1)
        std = torch.sigmoid(std)
        std = self.min_std + (self.max_std - self.min_std) * std
        return mean, std

    def _sample(self, mean, std):
        eps = torch.randn_like(std)
        return mean + eps * std

    def observe_step(self, obs_embed: torch.Tensor, action: torch.Tensor, state: dict):
        # obs_embed: [B, obs_dim], action: [B, action_dim]
        h = state["h"]
        z = state["z"]
        x = torch.cat([z, action], dim=-1)
        h = self.gru(x, h)
        post_params = self.post_net(torch.cat([h, obs_embed], dim=-1))
        post_mean, post_std = self._get_stats(post_params)
        z = self._sample(post_mean, post_std)
        return {"h": h, "z": z}

    def observe(self, obs_seq: torch.Tensor, action_seq: torch.Tensor):
        # obs_seq: [B, T, obs_dim], action_seq: [B, T, action_dim]
        B, T, _ = obs_seq.shape
        h = torch.zeros(B, self.deter_dim, device=obs_seq.device)
        z = torch.zeros(B, self.stoch_dim, device=obs_seq.device)

        hs, zs = [], []
        prior_means, prior_stds = [], []
        post_means, post_stds = [], []

        for t in range(T):
            a = action_seq[:, t]
            x = torch.cat([z, a], dim=-1)
            h = self.gru(x, h)

            prior_params = self.prior_net(h)
            prior_mean, prior_std = self._get_stats(prior_params)

            post_params = self.post_net(torch.cat([h, obs_seq[:, t]], dim=-1))
            post_mean, post_std = self._get_stats(post_params)
            z = self._sample(post_mean, post_std)

            hs.append(h)
            zs.append(z)
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            post_means.append(post_mean)
            post_stds.append(post_std)

        return {
            "h": torch.stack(hs, dim=1),
            "z": torch.stack(zs, dim=1),
            "prior_mean": torch.stack(prior_means, dim=1),
            "prior_std": torch.stack(prior_stds, dim=1),
            "post_mean": torch.stack(post_means, dim=1),
            "post_std": torch.stack(post_stds, dim=1),
        }

    def imagine(self, action_seq: torch.Tensor, start_state: dict):
        B, T, _ = action_seq.shape
        h = start_state["h"]
        z = start_state["z"]

        hs, zs = [], []
        prior_means, prior_stds = [], []

        for t in range(T):
            a = action_seq[:, t]
            x = torch.cat([z, a], dim=-1)
            h = self.gru(x, h)

            prior_params = self.prior_net(h)
            prior_mean, prior_std = self._get_stats(prior_params)
            z = self._sample(prior_mean, prior_std)

            hs.append(h)
            zs.append(z)
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)

        return {
            "h": torch.stack(hs, dim=1),
            "z": torch.stack(zs, dim=1),
            "prior_mean": torch.stack(prior_means, dim=1),
            "prior_std": torch.stack(prior_stds, dim=1),
        }
