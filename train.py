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

__Logger = logging.getLogger(__name__)

# "latent" with world model
# "standard" with default diffusion model settings
mode = "latent"

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
    def __init__(self, data_path = 'data/demo.npz'):
        data = np.load(data_path)
        self.obs = data["obs"]
        self.actions = data["actions"]
        self.obs_norm = Normalizer(self.obs)
        self.actions_norm = Normalizer(self.actions)
        self.norm_obs = self.obs_norm.normalize(self.obs)
        self.norm_actions = self.actions_norm.normalize(self.actions)
    
    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        n_obs = self.norm_obs[idx, 0, :]
        # action chunking, T = 16
        # sliding window might be needed
        n_actions = self.norm_actions[idx, :16, :]
        return {
            "cond": torch.FloatTensor(n_obs),
            "action": torch.FloatTensor(n_actions),
            "original_idx": idx,
        }

def train():
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    __Logger.info(f"Using device: {DEVICE}")

    dataset = RobotDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    network = ConditionalUnet1D(action_dim=dataset.actions.shape[2],
                                 obs_dim=dataset.obs.shape[2]).to(DEVICE)
    noise_scheduler = NoiseScheduler(num_timesteps=100, device=DEVICE)
    optimizer = torch.optim.AdamW(network.parameters(), lr=LR)

    loss_fn = nn.MSELoss()
    loss_history = []
    
    network.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0.0

        for batch in pbar:
            ncond = batch["cond"].to(DEVICE)
            naction = batch["action"].to(DEVICE)
            B = ncond.shape[0]
    
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (B,), device=DEVICE).long()
            
            noise = torch.randn_like(naction)

            noisy_actions = noise_scheduler.add_noise(
                naction, noise, timesteps)
            
            # input: noisy actions, timesteps, condition
            noise_pred = network(
                noisy_actions, timesteps, ncond)
            
            loss = loss_fn(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_epoch_loss)
    # save and visualization
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(network.state_dict(), "checkpoints/mini_diffusion.pth")
    __Logger.info("Model saved to checkpoints/mini_diffusion.pth")

    np.savez("checkpoints/normalization.npz",
             obs_min=dataset.obs_norm.min,
             obs_max=dataset.obs_norm.max,
             action_min=dataset.actions_norm.min,
             action_max=dataset.actions_norm.max)
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.savefig("checkpoints/training_loss.png")
    __Logger.info("Training loss plot saved to checkpoints/training_loss.png")

if __name__ == "__main__":
    train()