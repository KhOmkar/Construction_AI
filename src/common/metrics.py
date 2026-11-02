"""
Comprehensive metrics library for Construction Site AI:
- Reconstruction: MSE, PSNR, SSIM
- Anomaly Detection: AUROC, AUPRC, Optimal Threshold, FPR at 95% TPR, F1
- Generation Evaluation: Diversity, Nearest-Neighbor Novelty
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import integrate


def compute_reconstruction_metrics(real_tensor, recon_tensor):
    """
    Computes MSE, PSNR, and SSIM between batches of real and reconstructed images.
    Tensors should be (B, C, H, W) in [0, 1].
    """
    diff = (real_tensor - recon_tensor) ** 2
    mse = diff.mean().item()
    if mse < 1e-10:
        psnr = 100.0
    else:
        psnr = 10.0 * np.log10(1.0 / mse)

    # Simplified fast SSIM on PyTorch tensors
    ssim = compute_ssim_tensor(real_tensor, recon_tensor).item()

    return {
        "mse": float(mse),
        "psnr": float(psnr),
        "ssim": float(ssim)
    }


def compute_ssim_tensor(x, y, window_size=11, sigma=1.5):
    """Computes Mean Structural Similarity Index (SSIM) on torch tensors in [0, 1]."""
    channel = x.size(1)
    # Gaussian window
    gauss = torch.tensor([
        np.exp(-(i - window_size // 2) ** 2 / (2 * sigma ** 2))
        for i in range(window_size)
    ], dtype=x.dtype, device=x.device)
    gauss = gauss / gauss.sum()
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(x, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(y, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(x * x, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(y * y, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(x * y, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def compute_anomaly_metrics(scores, labels):
    """
    Vectorized computation of AUROC, AUPRC, FPR at 95% TPR, and optimal F1.
    scores: numpy array of anomaly scores (e.g. reconstruction error), higher = violation
    labels: binary numpy array (0 = compliant normal, 1 = violation anomaly)
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(int)

    # Sort descending by score
    desc_score_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[desc_score_indices]
    sorted_labels = labels[desc_score_indices]

    n_pos = np.sum(sorted_labels == 1)
    n_neg = np.sum(sorted_labels == 0)

    if n_pos == 0 or n_neg == 0:
        return {"auroc": 0.5, "auprc": 0.0, "fpr_at_95_tpr": 1.0, "optimal_threshold": 0.0, "best_f1": 0.0}

    # Cumulative True Positives and False Positives
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)

    tpr = tp / n_pos
    fpr = fp / n_neg
    precision = tp / (tp + fp)

    # AUROC via trapezoidal rule
    # Prepend (0,0) to ROC curve
    roc_fpr = np.concatenate(([0.0], fpr))
    roc_tpr = np.concatenate(([0.0], tpr))
    auroc = float(integrate.trapezoid(roc_tpr, roc_fpr))

    # AUPRC via trapezoidal rule
    pr_recall = np.concatenate(([0.0], tpr))
    pr_precision = np.concatenate(([1.0], precision))
    auprc = float(integrate.trapezoid(pr_precision, pr_recall))

    # FPR at 95% TPR
    idx_95 = np.searchsorted(tpr, 0.95)
    if idx_95 < len(fpr):
        fpr_95 = float(fpr[idx_95])
    else:
        fpr_95 = 1.0

    # Best F1 and optimal threshold
    f1_scores = 2 * (precision * tpr) / (precision + tpr + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_f1 = float(f1_scores[best_idx])
    optimal_thresh = float(sorted_scores[best_idx])

    return {
        "auroc": auroc,
        "auprc": auprc,
        "fpr_at_95_tpr": fpr_95,
        "best_f1": best_f1,
        "optimal_threshold": optimal_thresh
    }


def compute_diversity_lpips_batch(images):
    """
    Computes average pairwise L1/L2 diversity across generated sample images.
    images: (N, C, H, W) float tensor
    """
    n = images.size(0)
    if n < 2:
        return 0.0
    flat = images.reshape(n, -1)
    # Pairwise L2 distances
    dists = torch.cdist(flat, flat, p=2)
    # Mask out diagonal
    mask = ~torch.eye(n, dtype=bool, device=images.device)
    mean_dist = dists[mask].mean().item()
    return float(mean_dist)


def compute_nearest_neighbor_novelty(gen_images, train_images, sample_size=200):
    """
    Measures the distance from generated images to their nearest training neighbor.
    Verifies that the generative model is synthesizing novel images rather than memorizing training data.
    """
    with torch.no_grad():
        g_flat = gen_images.reshape(gen_images.size(0), -1)
        # Sample subset of train images for fast computation
        if train_images.size(0) > sample_size:
            idx = torch.randperm(train_images.size(0))[:sample_size]
            t_sub = train_images[idx]
        else:
            t_sub = train_images
        t_flat = t_sub.reshape(t_sub.size(0), -1)

        dists = torch.cdist(g_flat, t_flat, p=2)
        min_dists, _ = dists.min(dim=1)
        return float(min_dists.mean().item()), float(min_dists.min().item())
