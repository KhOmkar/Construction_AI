"""
Variational Autoencoder (VAE) and Class-Conditional VAE (C-VAE).
Features:
- Smooth latent space (Gaussian prior)
- Class conditioning (helmet, no_helmet, vest, no_vest)
- KL warmup schedule with configurable beta (0.5 to 4.0)
- Latent interpolation between worker states
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        latent_dim=128,
        num_classes=4,
        channels=(64, 128, 256, 512),
        class_emb_dim=32
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.class_emb_dim = class_emb_dim

        # Class embedding for encoder and decoder
        self.class_emb = nn.Embedding(num_classes, class_emb_dim)

        # Encoder conv blocks
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
        self.encoder_conv = nn.Sequential(*layers)

        self.flatten_dim = channels[-1] * 4 * 4  # 512 * 4 * 4 = 8192
        # Encoder projects image features + class embedding -> mu, logvar
        self.fc_mu = nn.Linear(self.flatten_dim + class_emb_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim + class_emb_dim, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim + class_emb_dim, channels[-1] * 4 * 4)
        self.start_shape = (channels[-1], 4, 4)

        rev_channels = list(reversed(channels))
        dec_layers = []
        for i in range(len(rev_channels) - 1):
            dec_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(rev_channels[i], rev_channels[i+1], kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(rev_channels[i+1]),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        dec_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(rev_channels[-1], in_channels, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            )
        )
        self.decoder_conv = nn.Sequential(*dec_layers)

    def encode(self, x, y):
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        y_emb = self.class_emb(y)
        h_cond = torch.cat([h, y_emb], dim=1)
        mu = self.fc_mu(h_cond)
        logvar = self.fc_logvar(h_cond)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y_emb = self.class_emb(y)
        z_cond = torch.cat([z, y_emb], dim=1)
        h = self.decoder_fc(z_cond)
        h = h.view(h.size(0), *self.start_shape)
        x_recon = self.decoder_conv(h)
        return x_recon

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, y)
        return x_recon, mu, logvar, z

    def compute_loss(self, x, x_recon, mu, logvar, beta=1.0):
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction="sum") / x.size(0)

        # Analytical KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        total_loss = recon_loss + beta * kl_loss

        return total_loss, {
            "loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "beta": beta
        }

    @torch.no_grad()
    def sample(self, num_samples, class_id, device="cuda"):
        self.eval()
        z = torch.randn(num_samples, self.latent_dim, device=device)
        y = torch.full((num_samples,), class_id, dtype=torch.long, device=device)
        return self.decode(z, y)

    @torch.no_grad()
    def interpolate(self, z1, z2, class_id, steps=10, device="cuda"):
        self.eval()
        alphas = torch.linspace(0.0, 1.0, steps, device=device).unsqueeze(1)
        z_interp = (1.0 - alphas) * z1.unsqueeze(0) + alphas * z2.unsqueeze(0)
        y = torch.full((steps,), class_id, dtype=torch.long, device=device)
        return self.decode(z_interp, y)
