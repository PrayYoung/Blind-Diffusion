import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import logging

from model.net import ConditionalUnet1D
from model.diffusion import NoiseScheduler
from model.world_model import SimpleWorldModel

__Logger = logging.getLogger(__name__)

# Need to run twice separately!!!
# "latent" with world model
# "standard" with default diffusion model settings
MODE = "latent"

BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Normalizer:
    def __init__(self, data):
        self.min = np.min(data, axis = (0,1))
        self.max = np.max(data, axis = (0,1))
        # prevent division by zero
        self.max[self.max == self.min] += 1e-8

    def normalize(self, x):
        norm = (x - self.min) / (self.max - self.min)
        # scale to [-1, 1]
        return 2 * norm - 1

    def denormalize(self, x):
        norm = (x + 1) / 2
        return norm * (self.max - self.min) + self.min

class RobotDataset(Dataset):
    def __init__(self, data_path = 'data/demo.npz', latents=None):
        data = np.load(data_path)
        self.obs = data["obs"]
        self.action = data["actions"]
        self.latents = latents

        self.obs_norm = Normalizer(self.obs)
        self.act_norm = Normalizer(self.action)
        self.norm_obs = self.obs_norm.normalize(self.obs)
        self.norm_action = self.act_norm.normalize(self.action)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        horizon = 16
        T = self.norm_action.shape[1]
        # random window
        max_start = T - horizon
        t = np.random.randint(0, max_start)
        # label
        target_action = self.norm_action[idx, t : t+horizon, :]

        if self.latents is not None:
            # get latents from memory
            cond = self.latents[idx, t]
        else:
            cond = self.norm_obs[idx, t]
        
        return {
            "cond": torch.FloatTensor(cond),
            "action": torch.FloatTensor(target_action)}

def precompute_latents(dataset_path = 'data/demo.npz'):
    __Logger.info("Precomputing latents for all trajectories...")
    data = np.load(dataset_path)
    obs = data["obs"]
    action = data["actions"]

    obs_norm = Normalizer(obs)
    act_norm = Normalizer(action)
    n_obs = torch.FloatTensor(obs_norm.normalize(obs))
    n_action = torch.FloatTensor(act_norm.normalize(action))

    wm = SimpleWorldModel(obs_dim=2, action_dim=2, hidden_dim=64).to(DEVICE)
    wm.load_state_dict(torch.load('checkpoints/world_model.pth', map_location=DEVICE))
    wm.eval()
    
    all_latents = []
    with torch.no_grad():
        for i in tqdm(range(len(n_obs))):
            curr_obs_seq = n_obs[i]
            curr_action_seq = n_action[i]
            prev_act_seq = torch.cat([torch.zeros(1,2).to(DEVICE),
                                      curr_action_seq[:-1]], dim=0)
            
            x = torch.cat([curr_obs_seq, prev_act_seq], dim=-1).unsqueeze(0)
            output, _ = wm.lstm(x)
            all_latents.append(output.squeeze(0).cpu().numpy())
    return np.array(all_latents)


def train():
    __Logger.info(f"Training in {MODE.upper()} mode on {DEVICE}")

    latents = None
    if MODE == "latent":
        latents = precompute_latents()

    dataset = RobotDataset(latents=latents)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    os.makedirs("checkpoints", exist_ok=True)
    np.savez("checkpoints/normalization.npz",
             obs_min=dataset.obs_norm.min,
             obs_max=dataset.obs_norm.max,
             action_min=dataset.act_norm.min,
             action_max=dataset.act_norm.max)
    
    obs_dim = 64 if MODE == "latent" else 2
    policy = ConditionalUnet1D(action_dim=2, obs_dim=obs_dim).to(DEVICE)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)
    noise_scheduler = NoiseScheduler(num_timesteps=100, device=DEVICE)
    loss_fn = nn.MSELoss()

    policy.train()
    loss_history = []

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            cond = batch["cond"].to(DEVICE)
            target = batch["action"].to(DEVICE)
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (cond.shape[0],), device=DEVICE).long()
            noise = torch.randn_like(target)
            # forward
            noisy_action = noise_scheduler.add_noise(
                target, noise, timesteps)
            # reverse
            pred_noise = policy(noisy_action, timesteps, cond)
            # loss & backprop
            loss = loss_fn(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        loss_history.append(epoch_loss / len(dataloader))

    save_name = f"checkpoints/mini_diffusion_{MODE}.pth"
    torch.save(policy.state_dict(), save_name)
    __Logger.info(f"Model saved to {save_name}")

    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.savefig(f"checkpoints/training_loss_{MODE}.png")

if __name__ == "__main__":
    train()


    
