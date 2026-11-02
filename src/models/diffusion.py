"""
Class-Conditional DDPM & DDIM Diffusion Model with Classifier-Free Guidance (CFG).
Uses diffusers.models.UNet2DModel sized specifically for Tier B 6GB VRAM:
- 64 base channels
- Spatial self-attention strictly at 16x16
- 10% label dropout for CFG
- DDIM 50-step deterministic sampler (with fast 20-step mode for Streamlit dashboard)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_diffusion_unet(
    input_resolution=64,
    in_channels=3,
    out_channels=3,
    base_channels=64,
    channel_mult=(1, 2, 2, 4),
    num_classes=4
):
    from diffusers import UNet2DModel

    block_channels = [base_channels * m for m in channel_mult]
    # Sized for 6 GB VRAM: Attention strictly at resolution 16x16 (index 2)
    down_block_types = [
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D", # 16x16
        "DownBlock2D"      # 8x8
    ]
    up_block_types = [
        "UpBlock2D",
        "AttnUpBlock2D",   # 16x16
        "UpBlock2D",
        "UpBlock2D"
    ]

    # num_class_embeds = 4 classes + 1 null/unconditional token = 5
    model = UNet2DModel(
        sample_size=input_resolution,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=2,
        block_out_channels=block_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        num_class_embeds=num_classes + 1,  # Index 4 is unconditional token
    )
    return model


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        unet,
        timesteps=1000,
        beta_schedule="cosine",
        num_classes=4,
        cfg_dropout_prob=0.10
    ):
        super().__init__()
        self.unet = unet
        self.timesteps = timesteps
        self.num_classes = num_classes
        self.null_class_id = num_classes  # Class token 4 represents unconditional
        self.cfg_dropout_prob = cfg_dropout_prob

        # Precompute diffusion schedules
        if beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, timesteps)
        elif beta_schedule == "cosine":
            # Nichol & Dhariwal cosine schedule
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers so they move with model.to(device)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, x, y):
        """Training forward pass: computes MSE loss predicting added noise."""
        b = x.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=x.device).long()
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x, t, noise)

        # Classifier-free guidance dropout during training
        mask = torch.rand(b, device=x.device) < self.cfg_dropout_prob
        y_train = y.clone()
        y_train[mask] = self.null_class_id

        noise_pred = self.unet(x_noisy, t, class_labels=y_train).sample
        loss = F.mse_loss(noise_pred, noise)
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def sample_ddim(
        self,
        num_samples,
        class_id,
        steps=50,
        guidance_scale=3.0,
        device="cuda"
    ):
        """
        Deterministic DDIM sampling with Classifier-Free Guidance.
        """
        self.eval()
        img = torch.randn((num_samples, 3, 64, 64), device=device)

        # Timestep subset for DDIM
        step_ratio = self.timesteps // steps
        timesteps_ddim = torch.arange(0, self.timesteps, step_ratio, device=device).flip(0)

        y_cond = torch.full((num_samples,), class_id, dtype=torch.long, device=device)
        y_uncond = torch.full((num_samples,), self.null_class_id, dtype=torch.long, device=device)

        denoising_progression = []

        for i, t in enumerate(timesteps_ddim):
            t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)

            # Predict conditional and unconditional noise
            noise_pred_cond = self.unet(img, t_batch, class_labels=y_cond).sample
            if guidance_scale > 1.0:
                noise_pred_uncond = self.unet(img, t_batch, class_labels=y_uncond).sample
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond

            alpha_cumprod_t = self.alphas_cumprod[t]
            alpha_cumprod_prev = self.alphas_cumprod[timesteps_ddim[i + 1]] if (i + 1) < len(timesteps_ddim) else torch.tensor(1.0, device=device)

            # Predict x_0
            pred_x0 = (img - torch.sqrt(1.0 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            # DDIM update to next step (eta = 0 for deterministic)
            dir_xt = torch.sqrt(1.0 - alpha_cumprod_prev) * noise_pred
            img = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt

            if i % max(1, steps // 5) == 0 or i == len(timesteps_ddim) - 1:
                denoising_progression.append(img.clamp(-1.0, 1.0))

        return img.clamp(-1.0, 1.0), denoising_progression
