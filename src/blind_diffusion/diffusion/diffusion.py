import torch
import torch.nn as nn
import torch.nn.functional as F

from .scheduler import cosine_beta_schedule, linear_beta_schedule


class GaussianDiffusion(nn.Module):
    """Gaussian diffusion supporting bounded direct-x0 prediction.

    Native RoboMimic actions live in [-1, 1].  Direct x0 prediction avoids
    the 1/sqrt(alpha_bar) amplification inherent in epsilon-to-x0 conversion
    at the terminal tail of a cosine schedule.  ``prediction_type='epsilon'``
    remains available for legacy and non-native-action configurations.
    """
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 100,
        schedule: str = "cosine",
        prediction_type: str = "epsilon",
    ):
        super().__init__()
        if prediction_type not in {"x0", "epsilon"}:
            raise ValueError(f"Unsupported prediction_type={prediction_type!r}")
        self.model = model
        self.timesteps = timesteps
        self.prediction_type = prediction_type

        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=alphas.dtype), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        # Derived schedule buffers are reconstructed from betas, so keep old
        # checkpoints (which predate them) loadable under strict=True.
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            persistent=False,
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x0: [B, C, H]
        sqrt_acp = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_acp * x0 + sqrt_om * noise

    def predict_x0_from_eps(self, x: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1)
        return (x - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar)

    def model_predictions(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return intrinsically bounded x0 and its corresponding epsilon."""
        raw = self.model(x, t, cond)
        if self.prediction_type == "x0":
            x0 = torch.tanh(raw)
            alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1)
            eps = (x - torch.sqrt(alpha_bar) * x0) / torch.sqrt(1.0 - alpha_bar)
        else:
            eps = raw
            x0 = self.predict_x0_from_eps(x, t, eps).clamp(-1.0, 1.0)
        return x0, eps

    def p_sample(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x0, _ = self.model_predictions(x, t, cond)
        beta = self.betas[t].view(-1, 1, 1)
        alpha = self.alphas[t].view(-1, 1, 1)
        alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1)
        alpha_bar_prev = self.alphas_cumprod_prev[t].view(-1, 1, 1)
        mean = (
            beta * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar) * x0
            + (1.0 - alpha_bar_prev) * torch.sqrt(alpha) / (1.0 - alpha_bar) * x
        )
        if (t == 0).all():
            return mean
        noise = torch.randn_like(x)
        variance = self.posterior_variance[t].view(-1, 1, 1)
        return mean + torch.sqrt(variance.clamp_min(0.0)) * noise

    def loss(self, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        if self.prediction_type == "x0":
            return F.mse_loss(torch.tanh(self.model(xt, t, cond)), x0)
        pred = self.model(xt, t, cond)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample_ddpm(self, shape, cond: torch.Tensor, return_trajectory: bool = False):
        x = torch.randn(shape, device=cond.device)
        trajectory = []
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=cond.device, dtype=torch.long)
            x = self.p_sample(x, t, cond)
            trajectory.append(float(x.detach().abs().max().cpu()))
        return (x, trajectory) if return_trajectory else x

    @torch.no_grad()
    def sample_ddim(self, shape, cond: torch.Tensor, inference_steps: int = 10, eta: float = 0.0, return_trajectory: bool = False):
        """Deterministic DDIM (eta=0) sampling over a strided timestep schedule."""
        if not 1 <= inference_steps <= self.timesteps:
            raise ValueError(f"inference_steps must be in [1, {self.timesteps}], got {inference_steps}")

        if eta < 0:
            raise ValueError("eta must be non-negative")
        x = torch.randn(shape, device=cond.device)
        trajectory = []
        schedule = torch.linspace(
            self.timesteps - 1, 0, inference_steps, device=cond.device
        ).round().long()
        for index, t_scalar in enumerate(schedule):
            t = torch.full((shape[0],), int(t_scalar.item()), device=cond.device, dtype=torch.long)
            x0, eps = self.model_predictions(x, t, cond)
            alpha_t = self.alphas_cumprod[t_scalar]
            alpha_prev = (
                self.alphas_cumprod[schedule[index + 1]]
                if index + 1 < len(schedule)
                else torch.ones((), device=cond.device, dtype=x.dtype)
            )
            sigma = eta * torch.sqrt((1.0 - alpha_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_prev))
            direction = torch.sqrt((1.0 - alpha_prev - sigma.square()).clamp_min(0.0)) * eps
            noise = sigma * torch.randn_like(x) if eta > 0 else 0.0
            x = torch.sqrt(alpha_prev) * x0 + direction + noise
            trajectory.append(float(x.detach().abs().max().cpu()))
        return (x, trajectory) if return_trajectory else x

    @torch.no_grad()
    def sample(self, shape, cond: torch.Tensor, sampler: str = "ddpm", inference_steps: int | None = None, eta: float = 0.0, return_trajectory: bool = False):
        if sampler == "ddpm":
            return self.sample_ddpm(shape, cond, return_trajectory)
        if sampler == "ddim":
            return self.sample_ddim(shape, cond, inference_steps or 10, eta, return_trajectory)
        raise ValueError(f"Unknown diffusion sampler: {sampler}")
