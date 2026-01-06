import torch
import numpy as np
import matplotlib.pyplot as plt
from env import ObstacleEnv
from model.net import ConditionalUnet1D
from model.diffusion import NoiseScheduler
from model.world_model import SimpleWorldModel
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

def run_blackout_experiment():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    stats = np.load('checkpoints/normalization.npz')
    obs_norm = Normalizer({"min": stats["obs_min"], "max": stats["obs_max"]})
    action_norm = Normalizer({"min": stats["action_min"], "max": stats["action_max"]})

    # load diffusion policy
    policy_std = ConditionalUnet1D(action_dim=2, obs_dim=2).to(DEVICE)
    policy_std.load_state_dict(torch.load('checkpoints/mini_diffusion_standard.pth',
                                          map_location=DEVICE))
    policy_std.eval()
    # load diffusion policy with world model (latent)
    policy_latent = ConditionalUnet1D(action_dim=2, obs_dim=64).to(DEVICE)
    policy_latent.load_state_dict(torch.load('checkpoints/mini_diffusion_latent.pth',
                                             map_location=DEVICE))
    policy_latent.eval()

    wm = SimpleWorldModel(obs_dim=2, action_dim=2, hidden_dim=64).to(DEVICE)
    wm.load_state_dict(torch.load('checkpoints/world_model.pth',
                                  map_location=DEVICE))
    wm.eval()

    noise_scheduler = NoiseScheduler(num_timesteps=100, device=DEVICE)
    env = ObstacleEnv()

    BLACKOUT_X_RANGE = [0, 0]
    # BLACKOUT_X_RANGE = [35, 65]

    plt.figure(figsize=(10, 8))
    circle = plt.Circle(env.obstacle_center, env.obstacle_radius, color='r', alpha=0.5)
    plt.gca().add_patch(circle)
    plt.scatter(env.start_pos[0], env.start_pos[1], color='g',
                s = 100, zorder = 5, label='Start')
    plt.scatter(env.target_pos[0], env.target_pos[1], color='orange',
                s = 100, zorder = 5, label='Target')
    # blackout area
    plt.axvspan(BLACKOUT_X_RANGE[0], BLACKOUT_X_RANGE[1], color='gray',
                alpha=0.3, label='Sensor Blackout Region')

    def run_sim(mode, color, label_prefix):
        __Logger.info(f"Running blackout simulation in {mode} mode...")
        for _ in range(1):
            curr_pos = env.start_pos.copy()
            history = [curr_pos.copy()]
            wm_state = None
            last_action = np.zeros(2)

            for step in range(0, 200, 8):
                is_blind = BLACKOUT_X_RANGE[0] <= curr_pos[0] <= BLACKOUT_X_RANGE[1]
                if is_blind:
                    obs_input = np.zeros(2)
                else:
                    obs_input = curr_pos

                n_obs = torch.FloatTensor(obs_norm.normalize(obs_input)).to(DEVICE).unsqueeze(0)
                if mode == "standard":
                    cond = n_obs
                    network = policy_std
                else:
                    n_act = torch.FloatTensor(action_norm.normalize(last_action)).to(DEVICE).unsqueeze(0)
                    with torch.no_grad():
                        # update the internal state
                        latent, wm_state = wm.get_latent(
                            n_obs.unsqueeze(1), 
                            n_act.unsqueeze(1), 
                            wm_state
                        )
                    cond = latent
                    network = policy_latent
                
                # diffusion generation
                noisy = torch.randn(1, 16, 2).to(DEVICE)
                for t in range(99, -1, -1):
                    ts = torch.tensor([t], device=DEVICE).long()
                    with torch.no_grad():                        
                        pred = network(noisy, ts, cond)
                        noisy = noise_scheduler.step(pred, t, noisy)
                actions = action_norm.denormalize(noisy.cpu().numpy()[0])
                # execution
                for j in range(8):
                    if step + j >= 200:
                        break
                    act = actions[j]
                    curr_pos = curr_pos + act
                    history.append(curr_pos.copy())
                    if mode == "latent":
                        last_action = act
                    if np.linalg.norm(curr_pos - env.target_pos) < 5:
                        break
                if np.linalg.norm(curr_pos - env.target_pos) < 5:
                    break
            history = np.array(history)
            plt.plot(history[:,0], history[:,1], color=color, linewidth=3,
                     label=f"{label_prefix} Policy Trajectory")

    run_sim(mode="standard", color="blue", label_prefix="Standard")
    run_sim(mode="latent", color="green", label_prefix="Latent")

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend(loc = "lower right")
    plt.title("Diffusion Policy with Sensor Blackout Experiment")
    plt.grid()
    plt.savefig("checkpoints/blackout_experiment.png")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_blackout_experiment()
