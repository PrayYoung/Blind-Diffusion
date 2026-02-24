import torch


class CEMPlanner:
    def __init__(self, horizon, action_dim, population, elites, iterations, alpha, action_low, action_high):
        self.horizon = horizon
        self.action_dim = action_dim
        self.population = population
        self.elites = elites
        self.iterations = iterations
        self.alpha = alpha
        self.action_low = action_low
        self.action_high = action_high

    def plan(self, start_state, model, terminal_penalty, constraint_penalty):
        device = start_state["h"].device
        mean = torch.zeros(self.horizon, self.action_dim, device=device)
        std = torch.ones(self.horizon, self.action_dim, device=device)

        for _ in range(self.iterations):
            actions = mean + std * torch.randn(self.population, self.horizon, self.action_dim, device=device)
            actions = actions.clamp(self.action_low, self.action_high)

            returns = self._evaluate(actions, start_state, model, terminal_penalty, constraint_penalty)
            elite_idx = torch.topk(returns, self.elites, dim=0).indices
            elite_actions = actions[elite_idx]
            new_mean = elite_actions.mean(dim=0)
            new_std = elite_actions.std(dim=0) + 1e-6
            mean = self.alpha * new_mean + (1 - self.alpha) * mean
            std = self.alpha * new_std + (1 - self.alpha) * std

        return mean[0]

    def _evaluate(self, actions, start_state, model, terminal_penalty, constraint_penalty):
        B = actions.shape[0]
        start = {"h": start_state["h"].repeat(B, 1), "z": start_state["z"].repeat(B, 1)}
        rollout = model.rssm.imagine(actions, start)
        feat = torch.cat([rollout["h"], rollout["z"]], dim=-1)
        reward = model.reward_head(feat)
        done = model.done_head(feat)
        obs_pred = model.obs_head(feat)
        from .cost import compute_return
        returns = compute_return(reward, done, obs_pred, terminal_penalty, constraint_penalty)
        return returns.sum(dim=1)
