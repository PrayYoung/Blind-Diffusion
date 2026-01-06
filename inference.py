import torch
import numpy as np
import matplotlib.pyplot as plt
from env import ObstacleEnv
from model.net import ConditionalUnet1D
from model.diffusion import NoiseScheduler
import logging

__Logger = logging.getLogger(__name__)

class Normalizer:
    def __init__(self, stats):
        self.min = stats['min']
        self.max = stats['max']

    def denormalize(self, x):
        # [-1, 1] to [min, max]
        return ((x + 1) /2) * (self.max - self.min) + self.min

    def normalize(self, x):
        return (x - self.min) / (self.max - self.min) * 2 - 1


def run_inference():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    stats = np.load('checkpoints/normalization.npz')
    obs_norm = Normalizer({"min": stats["obs_min"], "max": stats["obs_max"]})
    action_norm = Normalizer({"min": stats["action_min"], "max": stats["action_max"]})

    network = ConditionalUnet1D(action_dim=2, obs_dim=2, hidden_dim=64).to(DEVICE)
    network.load_state_dict(torch.load('checkpoints/mini_diffusion.pth', map_location=DEVICE))
    network.eval()

    noise_scheduler = NoiseScheduler(num_timesteps=100, device=DEVICE)


    env = ObstacleEnv()
    num_trials = 50
    start_obs = env.start_pos
    # input shape (B, obs_dim)
    cond = torch.FloatTensor(obs_norm.normalize(start_obs)).to(DEVICE)
    cond = cond.unsqueeze(0).repeat(num_trials, 1)

    # diffusion process
    T = 16
    noisy_action = torch.randn(num_trials, T, 2).to(DEVICE)
    __Logger.info("Diffusion Generation started...")
    for t in reversed(range(noise_scheduler.num_timesteps)):
        timesteps = torch.full((num_trials,), t, dtype=torch.long)
        with torch.no_grad():
            # predict noise
            pred_noise = network(noisy_action, timesteps, cond)
            # noise removal/ update step
            noisy_action = noise_scheduler.step(pred_noise, t, noisy_action)
    pred_actions = noisy_action.cpu().numpy()
    pred_actions = action_norm.denormalize(pred_actions)

    # visualization, need to integrate velocities to get positions
    plt.figure()
    circle = plt.Circle(env.obstacle_center, env.obstacle_radius, color='r', alpha=0.5)
    plt.gca().add_patch(circle)
    plt.scatter(env.start_pos[0], env.start_pos[1], color='g',
                s = 100, zorder = 5, label='Start')
    __Logger.info(f"Start Position: {env.start_pos}")
    for i in range(num_trials):
        traj = [env.start_pos]
        curr_pos = env.start_pos.copy()
        for t in range(T):
            action = pred_actions[i, t]
            curr_pos = curr_pos + action
            traj.append(curr_pos.copy())
        traj = np.array(traj)
        plt.plot(traj[:,0], traj[:,1], color = "blue", alpha=0.3)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title("Diffusion Policy Inference (Conditional on Start Position)")
    plt.savefig("checkpoints/diffusion_inference.png")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_inference()