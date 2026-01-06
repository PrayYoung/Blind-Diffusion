import torch
import torch.nn as nn

class SimpleWorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size=obs_dim + action_dim,
                            hidden_size=hidden_dim,
                            batch_first=True)
        self.predict_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, obs_dim)
        )

    def forward(self, x):
        """
        x: (B, T, obs_dim + action_dim)
        """
        output, _ = self.lstm(x)

        pred_next_obs = self.predict_head(output)
        return pred_next_obs, output
    
    def get_latent(self, obs, prev_action, h_prev=None):
        """
        Get latent representation from current obs and action
        obs: (B, 1, obs_dim)
        prev_action: (B, 1, action_dim)
        h_prev: (h_0, c_0), each of shape (1, B, hidden_dim)
        """
        x = torch.cat([obs, prev_action], dim=-1)
        output, h_current = self.lstm(x, h_prev)
        return output.squeeze(1), h_current
