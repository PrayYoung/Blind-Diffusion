import numpy as np
import matplotlib.pyplot as plt
from env import ObstacleEnv
import logging

Logger = logging.getLogger(__name__)

def visualize(max_trajectories: int = 50):
    with np.load("data/demo.npz", allow_pickle=False) as data:
        obs = data["obs"]  # shape (B,T,2)
        Logger.info(f"Loaded data with shape: {obs.shape}")

    env = ObstacleEnv()

    plt.figure(figsize=(8,8))

    circle = plt.Circle(env.obstacle_center, env.obstacle_radius, color='r',
                        alpha=0.5, label='Obstacle')
    plt.gca().add_patch(circle)

    num_to_plot = min(max_trajectories, obs.shape[0])
    for i in range(num_to_plot):
        plt.plot(obs[i,:,0], obs[i,:,1], color = 'blue', alpha=0.3)
    
    plt.scatter(env.start_pos[0], env.start_pos[1], color='green', s=100, label='Start')
    plt.scatter(env.target_pos[0], env.target_pos[1], color='orange', s=100, label='Target')

    plt.xlim(env.bounds)
    plt.ylim(env.bounds)
    plt.title("Expert Demonstrations (Multimodal Distribution)")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    visualize()
