"""
Vector-Quantized Variational Autoencoder (VQ-VAE) Image Tokenizer.
Tokenizes 64x64 images into a 16x16 grid of discrete tokens (512 codebook).
Provides the discrete vocabulary for the autoregressive transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = commitment_beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Uniform initialization
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        # z: (B, C, H, W)
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z_perm.view(-1, self.embedding_dim)

        # Distances: ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z.e
        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_z, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight).view(z_perm.shape)

        # Commitment loss
        loss = self.beta * F.mse_loss(quantized.detach(), z_perm) + F.mse_loss(quantized, z_perm.detach())

        # Straight-through estimator
        quantized = z_perm + (quantized - z_perm).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # Reshape indices to (B, H, W)
        indices = encoding_indices.view(z.shape[0], z.shape[2], z.shape[3])

        return quantized, loss, indices


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        codebook_size=512,
        embedding_dim=64,
        commitment_beta=0.25,
        hidden_channels=128
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim

        # Encoder: 64x64 -> 32x32 -> 16x16 (downsample by 4)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 2, 4, 2, 1), # 64 -> 32
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels // 2, hidden_channels, 4, 2, 1), # 32 -> 16
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, embedding_dim, 3, 1, 1) # 16x16 feature map
        )

        self.quantizer = VectorQuantizer(codebook_size, embedding_dim, commitment_beta)

        # Decoder: 16x16 -> 32x32 -> 64x64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden_channels, 4, 2, 1), # 16 -> 32
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, 4, 2, 1), # 32 -> 64
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels // 2, in_channels, 3, 1, 1),
            nn.Sigmoid() # Reconstruct in [0, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        quantized, vq_loss, indices = self.quantizer(h)
        return quantized, vq_loss, indices

    def decode(self, quantized):
        return self.decoder(quantized)

    def forward(self, x):
        quantized, vq_loss, indices = self.encode(x)
        x_recon = self.decode(quantized)
        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + vq_loss
        return x_recon, total_loss, {"loss": total_loss.item(), "recon_loss": recon_loss.item(), "vq_loss": vq_loss.item()}

    @torch.no_grad()
    def get_codebook_indices(self, x):
        h = self.encoder(x)
        _, _, indices = self.quantizer(h)
        # Flatten spatial 16x16 to (B, 256)
        return indices.view(x.size(0), -1)

    @torch.no_grad()
    def decode_indices(self, indices):
        # indices: (B, 256) in [0, codebook_size-1]
        b = indices.size(0)
        tokens = self.quantizer.embedding(indices) # (B, 256, embedding_dim)
        quantized = tokens.view(b, 16, 16, self.embedding_dim).permute(0, 3, 1, 2).contiguous()
        return self.decoder(quantized)
