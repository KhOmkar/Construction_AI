"""
Unified Cross-Model Benchmark Evaluator for Construction Site AI.
Evaluates all 5 generative architectures under an identical protocol:
- N = 2,048 synthetic crops per model (512 per safety class)
- Reference: Real test split (crops_test_64.npz, N = 2,524)
- Metrics: FID, KID, IS, CCA (Oracle Classifier), LPIPS Diversity, NN Novelty
- Output: results/all_runs.csv, results/tables/, results/figures/
"""

import os
import time
import json
import torch
import torchvision.utils as vutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from src.models.classifier import ReferenceClassifier, CropClassifier
from src.common.dataset import get_crop_dataloader
from src.common.metrics import (
    compute_diversity_lpips_batch,
    compute_nearest_neighbor_novelty
)


def load_real_test_features(test_npz_path, num_samples=2048, device="cuda"):
    data = np.load(test_npz_path)
    images = data["images"][:num_samples] # uint8 (N, 64, 64, 3)
    # Convert to contiguous float tensor (N, 3, 64, 64) in [0, 1]
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().float() / 255.0
    return tensor.to(device)


def evaluate_classifier_cca(classifier, gen_images, gen_labels, device="cuda"):
    """
    Computes Conditional Classification Accuracy (CCA).
    Measures percentage of synthetic images correctly recognized as their intended safety class.
    """
    classifier.eval()
    correct = 0
    total = len(gen_labels)
    batch_size = 128

    with torch.no_grad():
        for i in range(0, total, batch_size):
            b_imgs = gen_images[i:i+batch_size].to(device)
            b_lbls = gen_labels[i:i+batch_size].to(device)
            # Normalize to [-1, 1] as expected by ReferenceClassifier
            norm_imgs = (b_imgs - 0.5) / 0.5

            logits = classifier(norm_imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == b_lbls).sum().item()

    return float(correct / max(1, total))


def run_benchmark(device="cuda"):
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    print("\n" + "=" * 60)
    print("STARTING UNIFIED CROSS-MODEL BENCHMARK (N=2,048 samples/model)")
    print("=" * 60 + "\n")

    # 1. Load Reference Crop Classifier (Oracle)
    classifier = ReferenceClassifier(num_classes=4, in_channels=3).to(device)
    clf_ckpt = torch.load("checkpoints/classifier/best_classifier.pt", map_location=device, weights_only=False)
    classifier.load_state_dict(clf_ckpt["model_state_dict"])
    classifier.eval()
    print("Loaded Reference Oracle Classifier (Test Acc: 73.1%).")

    # 2. Load Real Test Images
    real_test_images = load_real_test_features("processed/crops/crops_test_64.npz", num_samples=2048, device=device)
    print(f"Loaded {len(real_test_images)} real test crops as reference benchmark.")

    # 3. Load Sample of Real Training Images for Novelty
    real_train_data = np.load("processed/crops/crops_train_64.npz")
    train_idx = np.random.choice(len(real_train_data["images"]), size=500, replace=False)
    real_train_sample = torch.from_numpy(real_train_data["images"][train_idx]).permute(0, 3, 1, 2).contiguous().float() / 255.0
    real_train_sample = real_train_sample.to(device)

    # 4. Metric evaluators (FID / KID)
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    real_uint8 = (real_test_images.clamp(0, 1) * 255).to(torch.uint8)

    models_config = [
        {"name": "Autoencoder", "type": "ae", "ckpt": "runs/ae_baseline/best.pt"},
        {"name": "Conditional VAE", "type": "cvae", "ckpt": "runs/cvae_baseline/best.pt"},
        {"name": "Conditional GAN", "type": "cgan", "ckpt": "runs/cgan_baseline/best.pt"},
        {"name": "DDPM Diffusion", "type": "diffusion", "ckpt": "runs/diffusion_baseline/best.pt"},
        {"name": "VQ-VAE + GPT", "type": "transformer", "ckpt": "runs/vqvae_transformer_baseline/best.pt"}
    ]

    benchmark_rows = []
    qualitative_samples = {}

    for m_info in models_config:
        m_name = m_info["name"]
        m_type = m_info["type"]
        ckpt_path = m_info["ckpt"]

        if not os.path.exists(ckpt_path):
            latest_path = ckpt_path.replace("best.pt", "latest.pt")
            if os.path.exists(latest_path):
                ckpt_path = latest_path
            else:
                print(f"Skipping {m_name}: checkpoint {ckpt_path} not found.")
                continue

        print(f"\n--- Evaluating Model: {m_name} ---")
        t0 = time.time()

        # Generate N=2,048 samples (512 per class)
        samples_per_class = 512
        gen_imgs_list = []
        gen_lbls_list = []
        grid_sample_crops = []

        if m_type == "ae":
            from src.models.ae import AnomalyAutoencoder
            model = AnomalyAutoencoder(in_channels=3, bottleneck_dim=256).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            # For Autoencoder, sample latent bottleneck z ~ N(0, I) and decode
            with torch.no_grad():
                for c in range(4):
                    z = torch.randn(samples_per_class, 256, device=device)
                    # Use decoder directly
                    imgs = model.decoder(z).clamp(0, 1)
                    gen_imgs_list.append(imgs)
                    gen_lbls_list.append(torch.full((samples_per_class,), c, dtype=torch.long, device=device))
                    grid_sample_crops.append(imgs[0].cpu())

        elif m_type == "cvae":
            from src.models.vae import ConditionalVAE
            model = ConditionalVAE(in_channels=3, latent_dim=128, num_classes=4, class_emb_dim=32).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            with torch.no_grad():
                for c in range(4):
                    z = torch.randn(samples_per_class, 128, device=device)
                    lbls = torch.full((samples_per_class,), c, dtype=torch.long, device=device)
                    imgs = model.decode(z, lbls).clamp(0, 1)
                    gen_imgs_list.append(imgs)
                    gen_lbls_list.append(lbls)
                    grid_sample_crops.append(imgs[0].cpu())

        elif m_type == "cgan":
            from src.models.gan import ConditionalGenerator
            model = ConditionalGenerator(latent_dim=128, num_classes=4).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["netG_state_dict"])
            model.eval()

            with torch.no_grad():
                for c in range(4):
                    z = torch.randn(samples_per_class, 128, device=device)
                    lbls = torch.full((samples_per_class,), c, dtype=torch.long, device=device)
                    imgs = model(z, lbls)
                    imgs = ((imgs + 1.0) * 0.5).clamp(0, 1)
                    gen_imgs_list.append(imgs)
                    gen_lbls_list.append(lbls)
                    grid_sample_crops.append(imgs[0].cpu())

        elif m_type == "diffusion":
            from src.models.diffusion import get_diffusion_unet, GaussianDiffusion
            unet = get_diffusion_unet(input_resolution=64, base_channels=64, num_classes=4).to(device)
            diffusion = GaussianDiffusion(unet=unet, num_classes=4).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "ema_unet_state_dict" in ckpt:
                unet.load_state_dict(ckpt["ema_unet_state_dict"])
            else:
                unet.load_state_dict(ckpt["unet_state_dict"])
            diffusion.eval()

            with torch.no_grad():
                for c in range(4):
                    # Fast DDIM 25 steps for benchmark throughput
                    imgs_c_list = []
                    for _ in range(samples_per_class // 64):
                        s, _ = diffusion.sample_ddim(num_samples=64, class_id=c, steps=25, guidance_scale=3.0, device=device)
                        imgs_c_list.append(((s + 1.0) * 0.5).clamp(0, 1))
                    imgs_c = torch.cat(imgs_c_list, dim=0)
                    gen_imgs_list.append(imgs_c)
                    gen_lbls_list.append(torch.full((len(imgs_c),), c, dtype=torch.long, device=device))
                    grid_sample_crops.append(imgs_c[0].cpu())

        elif m_type == "transformer":
            from src.models.vqvae import VQVAE
            from src.models.transformer import ImageTokenTransformer
            vqvae = VQVAE(in_channels=3, codebook_size=512, embedding_dim=64).to(device)
            gpt = ImageTokenTransformer(vocab_size=516, seq_len=257, n_layer=6, n_head=8, n_embd=384).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            vqvae.load_state_dict(ckpt["vqvae_state_dict"])
            gpt.load_state_dict(ckpt["gpt_state_dict"])
            vqvae.eval()
            gpt.eval()

            with torch.no_grad():
                for c in range(4):
                    imgs_c_list = []
                    for _ in range(samples_per_class // 64):
                        tokens = gpt.generate(class_id=c, num_samples=64, temperature=1.0, device=device)
                        crops = vqvae.decode_indices(tokens).clamp(0, 1)
                        imgs_c_list.append(crops)
                    imgs_c = torch.cat(imgs_c_list, dim=0)
                    gen_imgs_list.append(imgs_c)
                    gen_lbls_list.append(torch.full((len(imgs_c),), c, dtype=torch.long, device=device))
                    grid_sample_crops.append(imgs_c[0].cpu())

        all_gen_imgs = torch.cat(gen_imgs_list, dim=0)
        all_gen_lbls = torch.cat(gen_lbls_list, dim=0)
        qualitative_samples[m_name] = grid_sample_crops

        # 1. Compute FID & KID using torchmetrics
        print(f"Computing Inception FID / KID on {len(all_gen_imgs)} samples...")
        fid_metric = FrechetInceptionDistance(feature=64).to(device)
        kid_metric = KernelInceptionDistance(feature=64, subsets=10, subset_size=100).to(device)
        is_metric = InceptionScore().to(device)

        gen_uint8 = (all_gen_imgs.clamp(0, 1) * 255).to(torch.uint8)

        # Batch update to avoid VRAM exhaustion
        b_size = 128
        for i in range(0, len(real_uint8), b_size):
            fid_metric.update(real_uint8[i:i+b_size], real=True)
            kid_metric.update(real_uint8[i:i+b_size], real=True)

        for i in range(0, len(gen_uint8), b_size):
            fid_metric.update(gen_uint8[i:i+b_size], real=False)
            kid_metric.update(gen_uint8[i:i+b_size], real=False)
            is_metric.update(gen_uint8[i:i+b_size])

        fid_val = float(fid_metric.compute().item())
        kid_mean, kid_std = kid_metric.compute()
        kid_val = float(kid_mean.item())
        is_mean, is_std = is_metric.compute()
        is_val = float(is_mean.item())

        # 2. Compute CCA via Oracle Classifier
        cca_acc = evaluate_classifier_cca(classifier, all_gen_imgs, all_gen_lbls, device=device)

        # 3. Diversity (pairwise L2) & Novelty (NN to training set)
        diversity = compute_diversity_lpips_batch(all_gen_imgs[:200])
        nov_mean, nov_min = compute_nearest_neighbor_novelty(all_gen_imgs[:200], real_train_sample, sample_size=500)

        elapsed = time.time() - t0
        print(f"{m_name} Results -> FID: {fid_val:.2f} | KID: {kid_val:.4f} | IS: {is_val:.2f} | CCA: {cca_acc*100:.1f}% | Div: {diversity:.3f} | Nov: {nov_mean:.3f}")

        benchmark_rows.append({
            "Model": m_name,
            "Architecture": m_type.upper(),
            "FID": round(fid_val, 2),
            "KID": round(kid_val, 4),
            "Inception_Score": round(is_val, 2),
            "CCA_Accuracy": round(cca_acc * 100, 2),
            "Diversity_LPIPS": round(diversity, 4),
            "Novelty_NN_Dist": round(nov_mean, 4),
            "Novelty_Min_Dist": round(nov_min, 4),
            "Eval_Seconds": round(elapsed, 1)
        })

        # Free memory between models
        if 'model' in locals():
            del model
        if 'diffusion' in locals():
            del diffusion
        if 'unet' in locals():
            del unet
        if 'vqvae' in locals():
            del vqvae
        if 'gpt' in locals():
            del gpt
        torch.cuda.empty_cache()

    # 5. Save Leaderboard CSV & Tables
    df_results = pd.DataFrame(benchmark_rows)
    df_results.to_csv("results/all_runs.csv", index=False)
    print("\nBenchmark Ledger saved to results/all_runs.csv")

    # Save Markdown Table
    md_table = df_results.to_markdown(index=False)
    with open("results/tables/benchmark_table.md", "w") as f:
        f.write("# Cross-Model Generative Benchmark Leaderboard\n\n" + md_table + "\n")

    # Save LaTeX Table
    with open("results/tables/benchmark_table.tex", "w") as f:
        f.write(df_results.to_latex(index=False))

    # 6. Generate 4x5 Qualitative Comparison Grid
    class_names = ["Compliant", "No Helmet", "No Vest", "Both Missing"]
    model_names = list(qualitative_samples.keys())
    if model_names:
        fig, axes = plt.subplots(4, len(model_names), figsize=(3 * len(model_names), 12))
        for col_idx, m_name in enumerate(model_names):
            crops = qualitative_samples[m_name]
            for row_idx in range(4):
                ax = axes[row_idx, col_idx] if len(model_names) > 1 else axes[row_idx]
                ax.imshow(crops[row_idx].permute(1, 2, 0).numpy())
                ax.axis("off")
                if row_idx == 0:
                    ax.set_title(m_name, fontsize=12, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(class_names[row_idx], fontsize=11, fontweight="bold")

        plt.suptitle("Qualitative Cross-Model Safety Synthesis Comparison", fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig("results/figures/cross_model_comparison_grid.png", dpi=200, bbox_inches="tight")
        plt.close()

    print("\nBenchmark Evaluation Pipeline Complete!")
    return df_results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_benchmark(device=device)
