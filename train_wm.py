import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from model.world_model import SimpleWorldModel

__Logger = logging.getLogger(__name__)

class SequenceDataset(Dataset):
    def __init__(self, data_path='data/demo.npz'):
        data = np.load(data_path)
        self.obs = data["obs"] # (B, 100, 2)
        self.actions = data["actions"]
        self.obs_min, self.obs_max = self.obs.min(), self.obs.max()
        self.act_min, self.act_max = self.actions.min(), self.actions.max()
    
    def normalize(self, x, x_min, x_max):
        norm = (x - x_min) / (x_max - x_min + 1e-8)
        return 2 * norm - 1

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        # return the whole trajectory
        o = self.normalize(self.obs[idx], self.obs_min, self.obs_max)
        a = self.normalize(self.actions[idx], self.act_min, self.act_max)
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
    # padding val
    act_min = torch.tensor(dataset.act_min).to(DEVICE)
    act_max = torch.tensor(dataset.act_max).to(DEVICE)
    padding_val = 2 * (0 - act_min) / (act_max - act_min + 1e-8) - 1

    __Logger.info("Training World Model started...")
    for epoch in range(50):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/50")

        sampling_ratio = start_ratio + (end_ratio - start_ratio) * (epoch / 50)

        for obs_seq, act_seq in pbar:
            obs_seq = obs_seq.to(DEVICE)
            act_seq = act_seq.to(DEVICE)
            B, T, _ = obs_seq.shape

            # shift right by 1
            padding_tensor = torch.full((B,1,2), padding_val.item(), device = DEVICE)
            prev_act_full = torch.cat([padding_tensor, act_seq[:,:-1,:]],dim=1)
            # manual unrolling
            outputs = []
            last_pred_obs = None
            h = None

            for t in range(T):
                if t == 0:
                    current_obs = obs_seq[:, t:t+1,:]
                else:
                    current_obs = (last_pred_obs if torch.rand(1).item() < sampling_ratio
                                   else obs_seq[:,t:t+1,:])
                current_act = prev_act_full[:,t:t+1,:]
                
                x_step = torch.cat([current_obs, current_act], dim=-1)
                out_step , h = wm.lstm(x_step,h)
                pred_next = wm.predict_head(out_step)

                outputs.append(pred_next)
                last_pred_obs = pred_next
            
            outputs = torch.cat(outputs, dim=1)
            target = torch.cat([obs_seq[:,1:,:], obs_seq[:,-1:, :]], dim=1)

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
    torch.save(wm.state_dict(), "checkpoints/world_model.pth")
    __Logger.info("World Model saved to checkpoints/world_model.pth")

    plt.plot(loss_history)
    plt.title("World Model Training Loss")
    plt.savefig("checkpoints/world_model_loss.png")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_world_model()
