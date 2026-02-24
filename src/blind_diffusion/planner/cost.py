import torch


def constraint_violation_from_obs(obs: torch.Tensor, threshold: float = 5.0):
    # proxy: any large magnitude observation
    return (obs.abs() > threshold).any(dim=-1).float()


def compute_return(reward_pred: torch.Tensor, done_logits: torch.Tensor, obs_pred: torch.Tensor, terminal_penalty: float, constraint_penalty: float):
    done_prob = torch.sigmoid(done_logits)
    terminal_cost = done_prob * terminal_penalty
    violation = constraint_violation_from_obs(obs_pred)
    constraint_cost = violation * constraint_penalty
    return reward_pred - terminal_cost - constraint_cost
