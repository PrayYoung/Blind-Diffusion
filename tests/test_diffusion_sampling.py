import unittest

import torch
from torch import nn

from blind_diffusion.diffusion.diffusion import GaussianDiffusion


class ZeroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, t, cond):
        self.calls.append(t.detach().cpu())
        return torch.zeros_like(x)


class FixedX0Model(nn.Module):
    """Returns the pre-tanh logit for one known native-range action chunk."""
    def __init__(self, x0):
        super().__init__()
        self.register_buffer("logit_x0", torch.atanh(x0))

    def forward(self, x, t, cond):
        return self.logit_x0.expand_as(x)


class DiffusionSamplingTests(unittest.TestCase):
    def test_epsilon_reconstruction_matches_forward_process(self):
        diffusion = GaussianDiffusion(ZeroModel(), timesteps=20)
        x0 = torch.rand(3, 2, 4) * 1.5 - 0.75
        noise = torch.randn_like(x0)
        # Keep this algebraic epsilon oracle away from the terminal cosine
        # tail; terminal epsilon-to-x0 conditioning is tested separately by
        # the direct-x0 regression below.
        t = torch.tensor([0, 7, 15])
        xt = diffusion.q_sample(x0, t, noise)
        reconstructed = diffusion.predict_x0_from_eps(xt, t, noise)
        self.assertTrue(torch.allclose(reconstructed, x0, atol=1e-5))

    def test_ddim_is_deterministic_and_uses_requested_number_of_steps(self):
        model = ZeroModel()
        diffusion = GaussianDiffusion(model, timesteps=20)
        cond = torch.zeros(2, 3)

        torch.manual_seed(7)
        first = diffusion.sample((2, 2, 4), cond, sampler="ddim", inference_steps=5)
        self.assertEqual(len(model.calls), 5)
        self.assertLessEqual(float(first.abs().max()), 1.0)
        model.calls.clear()
        torch.manual_seed(7)
        second = diffusion.sample((2, 2, 4), cond, sampler="ddim", inference_steps=5)
        self.assertTrue(torch.allclose(first, second))
        self.assertEqual(len(model.calls), 5)

    def test_ddpm_remains_available(self):
        model = ZeroModel()
        diffusion = GaussianDiffusion(model, timesteps=6)
        out = diffusion.sample((1, 2, 4), torch.zeros(1, 3), sampler="ddpm")
        self.assertEqual(tuple(out.shape), (1, 2, 4))
        self.assertEqual(len(model.calls), 6)

    def test_direct_x0_loss_and_ddim_reconstruct_known_native_target(self):
        x0 = torch.tensor([[[0.25, -0.5, 0.75], [-0.2, 0.4, -0.6]]])
        diffusion = GaussianDiffusion(FixedX0Model(x0), timesteps=20, prediction_type="x0")
        self.assertLess(float(diffusion.loss(x0, torch.zeros(1, 3))), 1e-10)
        torch.manual_seed(9)
        ddim = diffusion.sample((1, 2, 3), torch.zeros(1, 3), sampler="ddim", inference_steps=5)
        self.assertTrue(torch.allclose(ddim, x0, atol=1e-5))
        self.assertLessEqual(float(ddim.abs().max()), 1.0)

    def test_direct_x0_ddpm_final_output_is_intrinsically_bounded(self):
        x0 = torch.tensor([[[0.9, -0.9], [0.5, -0.5]]])
        diffusion = GaussianDiffusion(FixedX0Model(x0), timesteps=20, prediction_type="x0")
        torch.manual_seed(3)
        ddpm = diffusion.sample((1, 2, 2), torch.zeros(1, 3), sampler="ddpm")
        self.assertTrue(torch.allclose(ddpm, x0, atol=1e-5))
        self.assertLessEqual(float(ddpm.abs().max()), 1.0)


if __name__ == "__main__":
    unittest.main()
