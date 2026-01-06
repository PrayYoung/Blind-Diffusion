import torch

class NoiseScheduler:
    def __init__(self, num_timesteps = 100, beta_start = 1e-4,
                 beta_end = 0.02, device = "cpu"):
        self.num_timesteps = num_timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alpha = 1.0 - self.betas
        # alpha_bar
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
    
    def add_noise(self, original_samples, noise, timesteps):
        """
        forward process: q(x_t | x_0)
        formula : x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        # shape (B, 1, 1)
        sqrt_alpha_prod = torch.sqrt(self.alpha_cumprod[timesteps]).flatten()
        sqrt_one_minus_alpha_prod = torch.sqrt(1 - self.alpha_cumprod[timesteps]).flatten()
        # reshape, sample shape: (B, T, D)
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = (sqrt_alpha_prod * original_samples +
                         sqrt_one_minus_alpha_prod * noise)
        return noisy_samples

    def step(self, model_output, timestep, sample):
        """
        reverse process: p(x_{t-1} | x_t)
        formula : x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * noise) + sigma_t * z
        """
        beta_t = self.betas[timestep]
        alpha_t = self.alpha[timestep]
        alpha_cumprod_t = self.alpha_cumprod[timestep]

        coeff1 = 1 / torch.sqrt(alpha_t)
        coeff2 = beta_t / torch.sqrt(1 - alpha_cumprod_t)
        pred_mean = coeff1 * (sample - coeff2 * model_output)

        # add variance (not for the last step)
        if timestep > 0:
            noise = torch.randn_like(sample)
            sigma_t = torch.sqrt(beta_t)
            prev_sample = pred_mean + sigma_t * noise
        else:
            prev_sample = pred_mean
        return prev_sample