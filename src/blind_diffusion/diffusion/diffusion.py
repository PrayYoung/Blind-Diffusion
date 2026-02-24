import torch
import torch.nn as nn
import torch.nn.functional as F

from .scheduler import cosine_beta_schedule, linear_beta_schedule


class GaussianDiffusion(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int = 100, schedule: str = "cosine"):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x0: [B, C, H]
        sqrt_acp = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_acp * x0 + sqrt_om * noise

    def p_sample(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        eps = self.model(x, t, cond)
        beta = self.betas[t].view(-1, 1, 1)
        alpha = self.alphas[t].view(-1, 1, 1)
        acp = self.alphas_cumprod[t].view(-1, 1, 1)
        mean = (1 / torch.sqrt(alpha)) * (x - beta / torch.sqrt(1 - acp) * eps)
        if (t == 0).all():
            return mean
        noise = torch.randn_like(x)
        return mean + torch.sqrt(beta) * noise

    def loss(self, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = self.model(xt, t, cond)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, shape, cond: torch.Tensor) -> torch.Tensor:
        x = torch.randn(shape, device=cond.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=cond.device, dtype=torch.long)
            x = self.p_sample(x, t, cond)
        return x
