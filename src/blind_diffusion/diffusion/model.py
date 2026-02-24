import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T], cond: [B, cond_dim]
        h = self.net(cond)
        scale, shift = torch.chunk(h, 2, dim=-1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        return x * (1 + scale) + shift


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.film = FiLM(cond_dim, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.film(h, cond)
        h = self.act(self.conv2(h))
        return h + self.skip(x)


class UNet1D(nn.Module):
    def __init__(self, act_dim: int, horizon: int, cond_dim: int, base_ch: int = 64):
        super().__init__()
        self.horizon = horizon
        self.time_emb = SinusoidalTimeEmbedding(base_ch)
        self.time_mlp = nn.Sequential(nn.Linear(base_ch, base_ch * 2), nn.SiLU(), nn.Linear(base_ch * 2, base_ch))

        self.down1 = ResBlock1D(act_dim, base_ch, cond_dim + base_ch)
        self.down2 = ResBlock1D(base_ch, base_ch * 2, cond_dim + base_ch)
        self.pool = nn.AvgPool1d(2)

        self.mid = ResBlock1D(base_ch * 2, base_ch * 2, cond_dim + base_ch)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.up1 = ResBlock1D(base_ch * 2, base_ch, cond_dim + base_ch)
        self.out = nn.Conv1d(base_ch, act_dim, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, act_dim, H], t: [B], cond: [B, cond_dim]
        t_emb = self.time_mlp(self.time_emb(t))
        cond_all = torch.cat([cond, t_emb], dim=-1)

        d1 = self.down1(x, cond_all)
        d2 = self.down2(self.pool(d1), cond_all)
        m = self.mid(d2, cond_all)
        u = self.up(m)
        u = self.up1(u, cond_all)
        return self.out(u)
