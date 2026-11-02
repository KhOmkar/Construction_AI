"""
Visualization helpers for generative models:
- Image grids from PyTorch tensors
- Anomaly reconstruction error heatmaps
- Latent space 2D projections (t-SNE / PCA / UMAP)
- Denoising progression strips
"""

import os
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def tensor_to_pil(tensor, normalize=True):
    """Convert a PyTorch tensor (C, H, W) or (B, C, H, W) to PIL Image."""
    if tensor.dim() == 4:
        grid = vutils.make_grid(tensor, normalize=normalize, scale_each=True, padding=2)
    else:
        grid = tensor
        if normalize:
            grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)

    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return Image.fromarray(ndarr)


def save_image_grid(tensor, out_path, nrow=8, normalize=True, value_range=(-1, 1)):
    """Save batch of images to a clean grid file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vutils.save_image(
        tensor,
        out_path,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range,
        padding=2
    )


def compute_reconstruction_heatmap(orig_tensor, recon_tensor):
    """
    Compute pixel-wise reconstruction error heatmap (L1 or L2)
    orig_tensor: (C, H, W) in [0, 1]
    recon_tensor: (C, H, W) in [0, 1]
    Returns: heatmap as uint8 numpy RGB array
    """
    diff = torch.abs(orig_tensor - recon_tensor).mean(dim=0).cpu().numpy()  # (H, W)
    # Normalize to [0, 1]
    diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

    # Use matplotlib colormap 'jet' or 'inferno'
    cmap = plt.get_cmap("inferno")
    heatmap_rgba = cmap(diff_norm)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    return heatmap_rgb, float(diff.mean())


def plot_reconstruction_trio(orig_img, recon_img, out_path, threshold=None, error_val=None):
    """
    Saves side-by-side: Original | Reconstruction | Error Heatmap
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Compute error heatmap
    heatmap, mean_err = compute_reconstruction_heatmap(orig_img, recon_img)

    orig_pil = tensor_to_pil(orig_img, normalize=False)
    recon_pil = tensor_to_pil(recon_img, normalize=False)
    heat_pil = Image.fromarray(heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(orig_pil)
    axes[0].set_title("Original Crop", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(recon_pil)
    axes[1].set_title("Reconstructed", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(heat_pil)
    title_str = f"Error Heatmap (L1: {mean_err:.4f})"
    if threshold is not None:
        verdict = "VIOLATION" if mean_err > threshold else "COMPLIANT"
        color = "red" if verdict == "VIOLATION" else "green"
        title_str += f"\n[{verdict}]"
        axes[2].set_title(title_str, fontsize=11, fontweight="bold", color=color)
    else:
        axes[2].set_title(title_str, fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
