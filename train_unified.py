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
from model.normalizer import Normalizer, load_normalization, save_normalization

__Logger = logging.getLogger(__name__)

# Need to run twice separately!!!
# "latent" with world model
# "standard" with default diffusion model settings
MODE = "standard"

BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RobotDataset(Dataset):
    def __init__(self, data_path = 'data/demo.npz', latents=None, obs_norm=None, act_norm=None):
        data = np.load(data_path)
        self.obs = data["obs"]
        self.action = data["actions"]
        self.latents = latents
        self.obs_norm = obs_norm or Normalizer.from_data(self.obs)
        self.act_norm = act_norm or Normalizer.from_data(self.action)
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
            # use previous-step latent to match inference
            if t == 0:
                cond = np.zeros_like(self.latents[idx, 0])
            else:
                cond = self.latents[idx, t - 1]
        else:
            cond = self.norm_obs[idx, t]
        
        return {
            "cond": torch.FloatTensor(cond),
            "action": torch.FloatTensor(target_action)}

def precompute_latents(dataset_path = 'data/demo.npz', obs_norm=None, act_norm=None):
    __Logger.info("Precomputing latents for all trajectories...")
    data = np.load(dataset_path)
    obs = data["obs"]
    action = data["actions"]

    obs_norm = obs_norm or Normalizer.from_data(obs)
    act_norm = act_norm or Normalizer.from_data(action)
    n_obs = torch.FloatTensor(obs_norm.normalize(obs)).to(DEVICE)
    n_action = torch.FloatTensor(act_norm.normalize(action)).to(DEVICE)

    wm = SimpleWorldModel(obs_dim=2, action_dim=2, hidden_dim=64).to(DEVICE)
    wm.load_state_dict(torch.load('checkpoints/world_model.pth', map_location=DEVICE))
    wm.eval()
    
    all_latents = []
    with torch.no_grad():
        for i in tqdm(range(len(n_obs))):
            curr_obs_seq = n_obs[i]
            curr_action_seq = n_action[i]
            x = torch.cat([curr_obs_seq, curr_action_seq], dim=-1).unsqueeze(0)
            output, _ = wm.lstm(x)
            all_latents.append(output.squeeze(0).cpu().numpy())
    return np.array(all_latents)


def train():
    __Logger.info(f"Training in {MODE.upper()} mode on {DEVICE}")

    os.makedirs("checkpoints", exist_ok=True)
    if os.path.exists("checkpoints/normalization.npz"):
        obs_norm, act_norm = load_normalization("checkpoints/normalization.npz")
    else:
        data = np.load("data/demo.npz")
        obs_norm = Normalizer.from_data(data["obs"])
        act_norm = Normalizer.from_data(data["actions"])
        save_normalization("checkpoints/normalization.npz", obs_norm, act_norm)

    latents = None
    if MODE == "latent":
        latents = precompute_latents(obs_norm=obs_norm, act_norm=act_norm)

    dataset = RobotDataset(latents=latents, obs_norm=obs_norm, act_norm=act_norm)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
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


    
