from .cem import CEMPlanner
from .mppi import MPPIPlanner


def build_planner(cfg, action_dim):
    p = cfg.planner
    if p.type == "cem":
        return CEMPlanner(p.horizon, action_dim, p.population, p.elites, p.iterations, p.alpha, p.action_low, p.action_high)
    if p.type == "mppi":
        return MPPIPlanner(p.horizon, action_dim, p.population, p.lambda_, p.noise_std, p.action_low, p.action_high)
    raise ValueError(f"Unknown planner: {p.type}")
