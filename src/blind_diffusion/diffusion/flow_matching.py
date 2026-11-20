"""Conditional flow matching with Lipman et al.'s Gaussian OT paths.

This uses independent Gaussian/action pairs, not minibatch OT assignment.
The learned vector field is unbounded; only completed action samples clamp.
"""
import torch
from torch import nn


class ConditionalOTFlowMatching(nn.Module):
    def __init__(self, model, sigma_min=1e-4, time_scale=1000.0):
        super().__init__()
        if not 0 <= sigma_min < 1 or time_scale <= 0:
            raise ValueError("Require 0<=sigma_min<1 and time_scale>0")
        self.model = model
        self.sigma_min = sigma_min
        # Retain continuous time but spread the shared sinusoidal embedding
        # over a useful frequency range. The ODE itself always uses t in [0,1].
        self.time_scale = time_scale

    def conditional_path(self, noise, target, time):
        if noise.shape != target.shape or time.shape != (target.shape[0],):
            raise ValueError("Require matching [B,C,H] endpoints and time [B]")
        t = time[:, None, None]
        state = (1 - (1 - self.sigma_min) * t) * noise + t * target
        velocity = target - (1 - self.sigma_min) * noise
        return state, velocity

    def velocity(self, state, time, condition):
        return self.model(state, time * self.time_scale, condition)

    def loss(self, target, cond, mask=None):
        if mask is not None:
            target = torch.where(mask.bool(), target, torch.zeros_like(target))
        noise = torch.randn_like(target)
        time = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        state, velocity = self.conditional_path(noise, target, time)
        error = (self.velocity(state, time, cond) - velocity).square()
        if mask is None:
            return error.mean()
        weight = mask.to(error.dtype).expand_as(error)
        return (error * weight).sum() / weight.sum().clamp_min(1)

    @torch.no_grad()
    def sample(self, shape, cond, sampler="euler", inference_steps=4,
               initial_noise=None, return_trajectory=False, clamp_final=True):
        if sampler != "euler" or not isinstance(inference_steps, int) or inference_steps < 1:
            raise ValueError("Flow sampling requires Euler and a positive integer step count")
        if initial_noise is not None and tuple(initial_noise.shape) != tuple(shape):
            raise ValueError("initial_noise shape does not match requested shape")
        state = (torch.randn(shape, device=cond.device, dtype=cond.dtype)
                 if initial_noise is None else initial_noise.to(cond).clone())
        trajectory = []
        for step in range(inference_steps):
            time = torch.full((shape[0],), step / inference_steps, device=cond.device, dtype=cond.dtype)
            state = state + self.velocity(state, time, cond) / inference_steps
            trajectory.append(float(state.abs().max()))
        # No tanh on velocity, and no clipping inside the ODE integration.
        result = state.clamp(-1, 1) if clamp_final else state
        return (result, trajectory) if return_trajectory else result
