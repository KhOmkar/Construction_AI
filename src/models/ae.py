"""
Convolutional Autoencoder for Construction Site Worker Anomaly Detection.
Trained exclusively on compliant workers (wearing both helmet and vest).
Reconstruction error flags safety violations (unhelmeted heads, missing vests).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common.metrics import compute_ssim_tensor


class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3, bottleneck_dim=256, channels=(64, 128, 256, 512)):
        super().__init__()
        layers = []
        prev_ch = in_channels
        for ch in channels:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(prev_ch, ch, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            prev_ch = ch
        self.conv_blocks = nn.Sequential(*layers)
        self.flatten_dim = channels[-1] * 4 * 4  # 512 * 4 * 4 = 8192
        self.fc = nn.Linear(self.flatten_dim, bottleneck_dim)

    def forward(self, x):
        h = self.conv_blocks(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, out_channels=3, bottleneck_dim=256, channels=(512, 256, 128, 64)):
        super().__init__()
        self.fc = nn.Linear(bottleneck_dim, channels[0] * 4 * 4)
        self.start_shape = (channels[0], 4, 4)

        layers = []
        for i in range(len(channels) - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        # Final block up to 64x64
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(channels[-1], out_channels, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid() # Reconstruct in [0, 1]
            )
        )
        self.deconv_blocks = nn.Sequential(*layers)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), *self.start_shape)
        x_recon = self.deconv_blocks(h)
        return x_recon


class AnomalyAutoencoder(nn.Module):
    def __init__(self, in_channels=3, bottleneck_dim=256, channels=(64, 128, 256, 512)):
        super().__init__()
        self.encoder = ConvEncoder(in_channels, bottleneck_dim, channels)
        self.decoder = ConvDecoder(in_channels, bottleneck_dim, list(reversed(channels)))
        self.bottleneck_dim = bottleneck_dim

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def compute_loss(self, x, x_recon, ssim_weight=0.20):
        mse_loss = F.mse_loss(x_recon, x)
        if ssim_weight > 0.0:
            ssim_val = compute_ssim_tensor(x, x_recon)
            loss = (1.0 - ssim_weight) * mse_loss + ssim_weight * (1.0 - ssim_val)
            return loss, {"loss": loss.item(), "mse": mse_loss.item(), "ssim": ssim_val.item()}
        return mse_loss, {"loss": mse_loss.item(), "mse": mse_loss.item()}

    def get_anomaly_scores(self, x):
        """
        Computes per-sample L1 reconstruction error as the anomaly score.
        Higher score = anomaly (PPE violation).
        """
        self.eval()
        with torch.no_grad():
            x_recon, _ = self.forward(x)
            # L1 pixel-wise error averaged across channels and spatial dimensions
            l1_diff = torch.abs(x - x_recon).mean(dim=[1, 2, 3])
            return l1_diff.cpu().numpy()
