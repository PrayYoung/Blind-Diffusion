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
    __Logger.info("Training World Model started...")
    for epoch in range(50):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/50")
        for obs_seq, act_seq in pbar:
            obs_seq = obs_seq.to(DEVICE)
            act_seq = act_seq.to(DEVICE)

            prev_act = torch.cat([torch.zeros_like(act_seq[:, :1, :]),
                                  act_seq[:, :-1, :]], dim=1)
            # input (B, T, 4)
            x = torch.cat([obs_seq, prev_act], dim=-1)
            # predict t+1 from t
            target = torch.cat([obs_seq[:, 1:, :], act_seq[:, -1:, :]], dim=1)
            pred_next, _ = wm(x)

            loss = loss_fn(pred_next, target)

            optimizer.zero_grad()
            loss.backward()
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
