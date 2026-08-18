import unittest

import torch

from blind_diffusion.models.rssm import RSSM


class RSSMTests(unittest.TestCase):
    def test_imagine_step_matches_imagine_for_one_step(self):
        rssm = RSSM(action_dim=2, obs_dim=5, deter_dim=7, stoch_dim=3, hidden_dim=11)
        state = {
            "h": torch.randn(4, 7),
            "z": torch.randn(4, 3),
        }
        action = torch.randn(4, 2)

        torch.manual_seed(7)
        one_step = rssm.imagine_step(action, state)
        torch.manual_seed(7)
        rollout = rssm.imagine(action.unsqueeze(1), state)

        self.assertEqual(one_step["h"].shape, (4, 7))
        self.assertEqual(one_step["z"].shape, (4, 3))
        self.assertTrue(torch.allclose(one_step["h"], rollout["h"][:, 0]))
        self.assertTrue(torch.allclose(one_step["z"], rollout["z"][:, 0]))
        self.assertTrue(torch.allclose(one_step["prior_mean"], rollout["prior_mean"][:, 0]))
        self.assertTrue(torch.allclose(one_step["prior_std"], rollout["prior_std"][:, 0]))

    def test_observe_step_preserves_causal_state(self):
        rssm = RSSM(action_dim=2, obs_dim=5, deter_dim=7, stoch_dim=3, hidden_dim=11)
        zero = {
            "h": torch.zeros(1, 7),
            "z": torch.zeros(1, 3),
        }
        obs0 = torch.randn(1, 5)
        obs1 = torch.randn(1, 5)
        action0 = torch.randn(1, 2)
        action1 = torch.randn(1, 2)

        torch.manual_seed(11)
        state0 = rssm.observe_step(obs0, action0, zero)
        torch.manual_seed(13)
        carried = rssm.observe_step(obs1, action1, state0)
        torch.manual_seed(13)
        reset = rssm.observe_step(obs1, action1, zero)

        self.assertFalse(torch.allclose(carried["h"], reset["h"]))
        self.assertFalse(torch.allclose(carried["z"], reset["z"]))


if __name__ == "__main__":
    unittest.main()
