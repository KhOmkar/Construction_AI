"""
Conditional GAN with Spectral Normalization and DiffAugment for Rare Violation Synthesis.
Synthesizes photorealistic workers with missing PPE (no_helmet, no_vest).
"""

import torch
import torch.nn as nn
from src.common.diffaugment import diff_augment


class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=128, num_classes=4, out_channels=3, base_channels=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.class_emb = nn.Embedding(num_classes, 32)
        in_dim = latent_dim + 32

        self.fc = nn.Linear(in_dim, base_channels * 8 * 4 * 4)
        self.base_ch = base_channels

        self.blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(base_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Output in [-1, 1]
        )

    def forward(self, z, y):
        y_emb = self.class_emb(y)
        h = torch.cat([z, y_emb], dim=1)
        h = self.fc(h)
        h = h.view(h.size(0), self.base_ch * 8, 4, 4)
        x = self.blocks(h)
        return x


class ConditionalDiscriminator(nn.Module):
    """
    Projection Discriminator (Miyato & Koyama) with Spectral Normalization.
    """
    def __init__(self, in_channels=3, num_classes=4, base_channels=64):
        super().__init__()
        sn = nn.utils.spectral_norm

        self.conv1 = sn(nn.Conv2d(in_channels, base_channels, 4, 2, 1)) # 64x64 -> 32x32
        self.conv2 = sn(nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)) # 32x32 -> 16x16
        self.conv3 = sn(nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)) # 16x16 -> 8x8
        self.conv4 = sn(nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1)) # 8x8 -> 4x4

        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.linear = sn(nn.Linear(base_channels * 8 * 4 * 4, 1))

        # Class embedding for projection discriminator
        self.class_emb = sn(nn.Embedding(num_classes, base_channels * 8 * 4 * 4))

    def forward(self, x, y, apply_diffaugment=True, policy="color,translation,cutout"):
        if apply_diffaugment and policy:
            x = diff_augment(x, policy=policy)

        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        h = self.act(self.conv4(h))
        h_flat = h.view(h.size(0), -1)

        # Unconditional score
        out = self.linear(h_flat).squeeze(1)

        # Projection conditioning: inner product between feature and class embedding
        w_y = self.class_emb(y)
        proj = torch.sum(h_flat * w_y, dim=1)

        return out + proj
