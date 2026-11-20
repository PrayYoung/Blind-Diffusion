import unittest

import torch

from blind_diffusion.diffusion.flow_matching import ConditionalOTFlowMatching
from blind_diffusion.diffusion.model import UNet1D
from blind_diffusion.diffusion.policy import build_action_policy


class LinearField(torch.nn.Module):
    def forward(self, state, time, cond):
        return state


class FlowMatchingTests(unittest.TestCase):
    def test_oracle_ot_field_reaches_target_with_expected_residual(self):
        class Oracle(torch.nn.Module):
            def forward(self, state, scaled_time, cond):
                time = scaled_time[:, None, None] / 1000.
                return (cond[:, :, None] - .99 * state) / (1 - .99 * time)
        flow = ConditionalOTFlowMatching(Oracle(), sigma_min=.01)
        noise = torch.randn(3, 2, 8)
        target = torch.tensor([[.4, -.7]]).expand(3, -1)
        for steps in (2, 4):
            sample = flow.sample(noise.shape, target, inference_steps=steps,
                                 initial_noise=noise, clamp_final=False)
            self.assertTrue(torch.allclose(sample, target[:, :, None] + .01 * noise, atol=1e-6))

    def test_path_endpoints_and_derivative(self):
        flow = ConditionalOTFlowMatching(LinearField(), sigma_min=.01)
        noise, target = torch.randn(3, 2, 8), torch.randn(3, 2, 8)
        start, v0 = flow.conditional_path(noise, target, torch.zeros(3))
        end, v1 = flow.conditional_path(noise, target, torch.ones(3))
        self.assertTrue(torch.equal(start, noise))
        self.assertTrue(torch.allclose(end, target + .01 * noise, atol=1e-6))
        self.assertTrue(torch.equal(v0, v1))
        self.assertTrue(torch.allclose(end - start, v0, atol=1e-6))

    def test_no_internal_clamping_and_exact_euler_steps(self):
        flow = ConditionalOTFlowMatching(LinearField())
        noise = torch.full((1, 1, 4), 2.)
        raw, trajectory = flow.sample(noise.shape, torch.zeros(1, 2), inference_steps=4,
            initial_noise=noise, return_trajectory=True, clamp_final=False)
        self.assertTrue(torch.allclose(raw, noise * 1.25 ** 4))
        self.assertEqual(len(trajectory), 4)
        self.assertGreater(trajectory[0], 1)
        bounded = flow.sample(noise.shape, torch.zeros(1, 2), inference_steps=4, initial_noise=noise)
        self.assertTrue(torch.equal(bounded, torch.ones_like(noise)))
        self.assertTrue(torch.equal(noise, torch.full_like(noise, 2.)))

    def test_masked_loss_and_gradients(self):
        flow = ConditionalOTFlowMatching(UNet1D(2, 8, 3, 8, architecture="skip_v2"))
        data, cond = torch.randn(2, 2, 8), torch.randn(2, 3)
        mask = torch.ones(2, 1, 8, dtype=torch.bool)
        mask[:, :, -2:] = False
        changed = data.clone()
        changed[:, :, -2:] = 100
        torch.manual_seed(1)
        first = flow.loss(data, cond, mask)
        torch.manual_seed(1)
        second = flow.loss(changed, cond, mask)
        self.assertTrue(torch.equal(first, second))
        first.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in flow.parameters() if p.grad is not None))

    def test_checkpoint_factory_and_continuous_time(self):
        config = {"policy_type": "flow_matching", "flow": {"sigma_min": .001, "time_scale": 100.},
            "diffusion": {"horizon": 8, "base_ch": 8, "architecture": "skip_v2"}}
        flow = build_action_policy(config, 2, 3)
        restored = build_action_policy(config, 2, 3)
        restored.load_state_dict(flow.state_dict())
        noise, cond = torch.randn(1, 2, 8), torch.randn(1, 3)
        self.assertTrue(torch.equal(flow.sample(noise.shape, cond, initial_noise=noise),
                                   restored.sample(noise.shape, cond, initial_noise=noise)))
        self.assertFalse(torch.equal(flow.velocity(noise, torch.tensor([.1]), cond),
                                     flow.velocity(noise, torch.tensor([.2]), cond)))
        with self.assertRaises(ValueError):
            flow.sample(noise.shape, cond, inference_steps=0)


if __name__ == "__main__":
    unittest.main()
