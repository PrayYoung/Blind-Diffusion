import torch
import torch.nn as nn


class SmallResNet(nn.Module):
    def __init__(self, in_ch: int = 3, embed_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.block1 = self._block(32, 64)
        self.block2 = self._block(64, 128)
        self.block3 = self._block(128, 256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, embed_dim)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
