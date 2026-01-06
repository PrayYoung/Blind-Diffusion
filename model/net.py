import torch
import torch.nn as nn
import math
import logging

__Logger = logging.getLogger(__name__)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConditionalUnet1D(nn.Module):
    def __init__(self, action_dim, obs_dim, hidden_dim = 64):
        super().__init__()
        # time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # observation embedding MLP
        self.obs_mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # down sampling / encoder
        self.down1 = nn.Sequential(
            nn.Conv1d(action_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.Mish(),
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.Mish(),
        )
        # up sampling / decoder
        self.up1 = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.Mish(),
        )
        self.final_conv = nn.Conv1d(hidden_dim, action_dim, kernel_size=1)
    
    def forward(self, sample, timestep, global_cond):
        """
        samples : (B, T, action_dim)
        timestep : (B,)
        global_cond : (B, obs_dim)
        """
        # swap dim, (B,T,C) --> (B,C,T)
        x = sample.permute(0, 2, 1)
        # embeddings
        t_emb = self.time_mlp(timestep)
        g_emb = self.obs_mlp(global_cond)

        # concatenate t_emb and obs_emb, reshape to (B, C, 1)
        cond = (t_emb + g_emb).unsqueeze(-1)
        # down sample
        x1 = self.down1(x)
        # Feature-wise linear modulation (FiLM)
        x1 = x1 + cond
        x2 = self.down2(x1)

        # up sample
        x_up = self.up1(x2)
        x_up = x_up + x1  # skip connection

        # final
        out = self.final_conv(x_up)
        out = out.permute(0, 2, 1)
        return out

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # sanity check
    # B = 2, T = 16, action_dim = 2, obs_dim = 2
    
    batch_size = 2
    T = 16
    act_dim = 2
    obs_dim = 2

    net = ConditionalUnet1D(action_dim=act_dim, obs_dim=obs_dim)
    noisy_action = torch.randn(batch_size, T, act_dim)
    timesteps = torch.randint(0, 100, (batch_size,))
    obs = torch.randn(batch_size, obs_dim)

    pred_noise = net(noisy_action, timesteps, obs)
    __Logger.info(f"Input shape: {noisy_action.shape}")
    __Logger.info(f"Output shape: {pred_noise.shape}")

    if noisy_action.shape != pred_noise.shape:
        __Logger.error("Output shape does not match input shape! Sanity check failed!")
    else:
        __Logger.info("Output shape matches input shape. Sanity check passed!")