import torch


class MPPIPlanner:
    def __init__(self, horizon, action_dim, population, lambda_, noise_std, action_low, action_high):
        self.horizon = horizon
        self.action_dim = action_dim
        self.population = population
        self.lambda_ = lambda_
        self.noise_std = noise_std
        self.action_low = action_low
        self.action_high = action_high
        self.action_seq = torch.zeros(horizon, action_dim)

    def plan(self, start_state, model, terminal_penalty, constraint_penalty):
        device = start_state["h"].device
        if self.action_seq.device != device:
            self.action_seq = self.action_seq.to(device)
        noise = self.noise_std * torch.randn(self.population, self.horizon, self.action_dim, device=device)
        actions = self.action_seq.unsqueeze(0) + noise
        actions = actions.clamp(self.action_low, self.action_high)

        returns = self._evaluate(actions, start_state, model, terminal_penalty, constraint_penalty)
        weights = torch.softmax(returns / self.lambda_, dim=0)
        self.action_seq = (weights[:, None, None] * actions).sum(dim=0)
        return self.action_seq[0]

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
