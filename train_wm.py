import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from model.world_model import SimpleWorldModel
from model.normalizer import Normalizer, save_normalization

__Logger = logging.getLogger(__name__)

class SequenceDataset(Dataset):
    def __init__(self, data_path='data/demo.npz'):
        data = np.load(data_path)
        self.obs = data["obs"] # (B, 100, 2)
        self.actions = data["actions"]
        self.obs_norm = Normalizer.from_data(self.obs)
        self.act_norm = Normalizer.from_data(self.actions)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        # return the whole trajectory
        o = self.obs_norm.normalize(self.obs[idx])
        a = self.act_norm.normalize(self.actions[idx])
        return torch.FloatTensor(o), torch.FloatTensor(a)

def train_world_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SequenceDataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # input: obs(2) + act(2)
    wm = SimpleWorldModel(obs_dim=2, action_dim=2, hidden_dim=64).to(DEVICE)
    optimizer = torch.optim.Adam(wm.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    loss_history = []

    # schedule sampling
    start_ratio = 0
    end_ratio = 0.25

    __Logger.info("Training World Model started...")
    for epoch in range(50):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/50")

        sampling_ratio = start_ratio + (end_ratio - start_ratio) * (epoch / 50)

        for obs_seq, act_seq in pbar:
            obs_seq = obs_seq.to(DEVICE)
            act_seq = act_seq.to(DEVICE)
            B, T, _ = obs_seq.shape

            # manual unrolling
            outputs = []
            last_pred_obs = None
            h = None

            for t in range(T - 1):
                if t == 0:
                    current_obs = obs_seq[:, t:t+1,:]
                else:
                    current_obs = (last_pred_obs if torch.rand(1).item() < sampling_ratio
                                   else obs_seq[:,t:t+1,:])
                current_act = act_seq[:,t:t+1,:]
                
                x_step = torch.cat([current_obs, current_act], dim=-1)
                out_step , h = wm.lstm(x_step,h)
                pred_next = wm.predict_head(out_step)

                outputs.append(pred_next)
                last_pred_obs = pred_next
            
            outputs = torch.cat(outputs, dim=1)
            target = obs_seq[:,1:,:]

            loss = loss_fn(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * obs_seq.size(0)
            pbar.set_postfix({"Loss": loss.item()})
        loss_history.append(epoch_loss / len(dataset))
    
    os.makedirs("checkpoints", exist_ok=True)
    save_normalization("checkpoints/normalization.npz",
                       dataset.obs_norm, dataset.act_norm)
    torch.save(wm.state_dict(), "checkpoints/world_model.pth")
    __Logger.info("World Model saved to checkpoints/world_model.pth")

    plt.plot(loss_history)
    plt.title("World Model Training Loss")
    plt.savefig("checkpoints/world_model_loss.png")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_world_model()
