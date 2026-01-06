import numpy as np
import os
import logging

__Logger = logging.getLogger(__name__)

class ObstacleEnv:
    def __init__(self):
        self.bounds = [0, 100]
        self.start_pos = np.array([10,50])
        self.target_pos = np.array([90,50])
        self.obstacle_center = np.array([50,50])
        self.obstacle_radius = 15
    
    def get_expert_trajectory(self, mode = "top", noise_level = 1.0):
        """Generate an expert trajectory avoiding the obstacle.
        Args:
            mode (str): "top" or "bottom" to choose the path around the obstacle.
            noise_level (float): Standard deviation of Gaussian noise to add to the trajectory.
        """
        num_steps = 100
        t = np.linspace(0, 1, num_steps)

        x = np.linspace(self.start_pos[0], self.target_pos[0], num_steps)
        direction = 1 if mode == "top" else -1
        mid_y_offset = 30 * direction
        base_y = self.start_pos[1] + mid_y_offset * np.sin(np.pi * t)

        noise_x = np.random.normal(0, noise_level, num_steps)
        noise_y = np.random.normal(0, noise_level, num_steps)

        window = 5
        noise_x = np.convolve(noise_x, np.ones(window)/window, mode="same")
        noise_y = np.convolve(noise_y, np.ones(window)/window, mode="same")

        traj_x = x + noise_x
        traj_y = base_y + noise_y

        observations = np.stack([traj_x, traj_y], axis=1)

        actions = np.zeros_like(observations)
        actions[:-1] = observations[1:] - observations[:-1]

        return observations, actions

def generate_dataset(num_episodes = 1000, save_path = "data/demo.npz", noise_level = 1.0):
    env = ObstacleEnv()
    all_obs = []
    all_actions = []

    __Logger.info(f"Generating dataset, {num_episodes} trajectories...")

    for _ in range(num_episodes):
        mode = "top" if np.random.rand() < 0.5 else "bottom"
        obs, actions = env.get_expert_trajectory(mode, noise_level)
        all_obs.append(obs)
        all_actions.append(actions)
    
    # shape (B,T,2)
    all_obs = np.array(all_obs, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.float32)

    # normalization needed
    stats = {
        "obs_min": np.min(all_obs, axis=(0,1)),
        "obs_max": np.max(all_obs, axis=(0,1)),
        "action_mean": np.mean(all_actions, axis=(0,1)),
        "action_std": np.std(all_actions, axis=(0,1)),
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.savez(save_path,obs=all_obs, actions=all_actions, **stats)
    __Logger.info(f"Dataset saved to {save_path}")
    __Logger.info(f"Obs shape: {all_obs.shape}, Actions shape: {all_actions.shape}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_dataset()
