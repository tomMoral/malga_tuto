import torch
import torch.nn as nn
import torch.nn.functional as F


class DIPSkipNet(nn.Module):
    """U-Net-style image generator for Deep Image Prior."""

    def __init__(self, in_channels=32, out_channels=3):
        super().__init__()
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
        )
        self.down = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.mid = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.up2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.up1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1), nn.ReLU()
        )
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, z):
        skip = self.skip(z)
        d = self.down(skip)
        m = self.mid(d)
        u2 = F.interpolate(
            m, scale_factor=2, mode="bilinear", align_corners=False
        )
        u2 = self.up2(u2)
        u1 = F.interpolate(
            u2, scale_factor=2, mode="bilinear", align_corners=False
        )
        u1 = self.up1(torch.cat([u1, skip], dim=1))
        return torch.sigmoid(self.final(u1))


class KernelNet(nn.Module):
    """Small MLP for blind kernel estimation."""

    def __init__(self, z_dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, kernel_size * kernel_size),
        )

    def forward(self, z):
        k = self.net(z).view(1, 1, self.kernel_size, self.kernel_size)
        k = F.softplus(k)
        return k / (k.sum() + 1e-8)
