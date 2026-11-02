"""
Generate Benchmark Tables and Visual Comparison Grid for Construction Site AI.
"""

import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def main():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # 1. Format Tables
    df = pd.read_csv("results/all_runs.csv")
    print("Loaded Benchmark CSV:\n", df)

    # Format Markdown
    md_table = df.to_markdown(index=False)
    with open("results/tables/benchmark_table.md", "w", encoding="utf-8") as f:
        f.write("# Cross-Model Generative Benchmark Leaderboard\n\n")
        f.write("Evaluated on held-out real test crops ($N = 2,048$ samples per architecture):\n\n")
        f.write(md_table + "\n")

    # Format LaTeX
    with open("results/tables/benchmark_table.tex", "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False))

    print("Saved benchmark_table.md and benchmark_table.tex")

    # 2. Extract representative crops from 4x4 sample grids
    # Each grid (C-VAE, CGAN, Diffusion, GPT) is 4 rows x 4 cols (nrow=4).
    # Row 0: Compliant (class 0)
    # Row 1: No Helmet (class 1)
    # Row 2: No Vest (class 2)
    # Row 3: Both Missing (class 3)

    def extract_rows_from_grid(img_path):
        grid = Image.open(img_path).convert("RGB")
        w, h = grid.size
        # Border/padding detection
        cw = w // 4
        ch = h // 4
        crops = []
        for r in range(4):
            # Take column 0 from row r
            crop = grid.crop((0, r * ch, cw, (r + 1) * ch))
            crops.append(crop)
        return crops

    # Autoencoder trios (take middle reconstruction from each trio)
    ae_crops = []
    trio_files = [
        "runs/ae_baseline/anomaly_trios/trio_compliant.png",
        "runs/ae_baseline/anomaly_trios/trio_no_helmet.png",
        "runs/ae_baseline/anomaly_trios/trio_no_vest.png",
        "runs/ae_baseline/anomaly_trios/trio_both_missing.png"
    ]
    for tf in trio_files:
        if os.path.exists(tf):
            t_img = Image.open(tf).convert("RGB")
            tw, th = t_img.size
            # middle crop (Reconstruction) is from tw/3 to 2*tw/3
            rec = t_img.crop((tw // 3, 0, (2 * tw) // 3, th))
            ae_crops.append(rec)
        else:
            ae_crops.append(Image.new("RGB", (64, 64), (128, 128, 128)))

    cvae_crops = extract_rows_from_grid("runs/cvae_baseline/samples/sample_epoch_100.png")
    cgan_crops = extract_rows_from_grid("runs/cgan_baseline/samples/sample_epoch_100.png")
    diff_crops = extract_rows_from_grid("runs/diffusion_baseline/samples/sample_epoch_100.png")
    gpt_crops = extract_rows_from_grid("runs/vqvae_transformer_baseline/samples/gpt_sample_epoch_100.png")

    columns = [
        ("Autoencoder\n(One-Class)", ae_crops),
        ("Conditional VAE\n(Latent Manifold)", cvae_crops),
        ("Conditional GAN\n(DiffAugment)", cgan_crops),
        ("DDPM Diffusion\n(CFG 3.0 DDIM)", diff_crops),
        ("VQ-VAE + GPT\n(Discrete Prior)", gpt_crops),
    ]

    class_labels = [
        "Class 0:\nCompliant\n(Helmet + Vest)",
        "Class 1:\nNo Helmet\n(Head Exposed)",
        "Class 2:\nNo Vest\n(Civilian Shirt)",
        "Class 3:\nBoth Missing\n(Full Violation)"
    ]

    # Create 4 rows x 5 columns grid
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    plt.subplots_adjust(wspace=0.08, hspace=0.15)

    for col_idx, (col_name, crops) in enumerate(columns):
        for row_idx in range(4):
            ax = axes[row_idx, col_idx]
            ax.imshow(crops[row_idx])
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(col_name, fontsize=12, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(class_labels[row_idx], fontsize=11, fontweight="bold", labelpad=8)

    plt.suptitle("Qualitative Cross-Model Safety Synthesis Matrix (100 Epochs Local Training)", fontsize=16, fontweight="bold", y=0.98)
    fig_path = "results/figures/cross_model_comparison_grid.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved qualitative grid to {fig_path}")

    # Copy to artifact directory
    artifact_dest = r"C:\Users\omkar\.gemini\antigravity\brain\2f39b089-5fd9-40e1-a4a9-72f8c3145641\cross_model_comparison_grid.png"
    shutil.copy2(fig_path, artifact_dest)
    print(f"Copied to artifact directory: {artifact_dest}")

if __name__ == "__main__":
    main()
