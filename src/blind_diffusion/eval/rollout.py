import torch


def rollout_policy(env, policy, episodes: int, render: bool = False):
    results = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        success = 0.0
        collision = 0.0
        while not done:
            if isinstance(obs, dict):
                obs_vec = torch.cat([torch.tensor(obs[k]).float() for k in obs.keys()], dim=-1)
            else:
                obs_vec = torch.tensor(obs).float()
            action = policy(obs_vec).detach().cpu().numpy()
            obs, reward, done, info = env.step(action)
            if render:
                env.render()
            if info.get("success", False):
                success = 1.0
            if info.get("collision", False) or info.get("violation", False):
                collision = 1.0
        results.append({"success": success, "collision": collision})
    return results
