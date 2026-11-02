"""
Build notebooks/00_data_overview.ipynb with complete structure, charts, and sample grids.
"""

import os
import json


def create_overview_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Notebook 00: Dataset Overview & Data Preparation\n",
                    "**Project:** Construction Site AI — PPE & Hazard Generative Modeling  \n",
                    "**Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (Tier B)  \n",
                    "**Date:** 2026-11-01  \n",
                    "\n",
                    "---\n",
                    "\n",
                    "## 1. Project Goal & Overview\n",
                    "This project implements five generative architectures (Autoencoder, VAE, Conditional GAN, Class-Conditional DDPM Diffusion, and VQ-VAE + Autoregressive Transformer) to monitor worker safety and synthesize rare PPE violation states (*no helmet*, *no vest*) on construction job sites.\n",
                    "\n",
                    "This notebook presents the Phase 1 unified dataset, deduplication statistics, split breakdowns, and 64x64 worker crop distributions."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import yaml\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from PIL import Image\n",
                    "\n",
                    "# Set working directory to project root\n",
                    "if not os.path.exists('configs') and os.path.exists('../configs'):\n",
                    "    os.chdir('..')\n",
                    "\n",
                    "with open('configs/dataset.yaml', 'r') as f:\n",
                    "    cfg = yaml.safe_load(f)\n",
                    "\n",
                    "print(f\"Project root: {os.getcwd()}\")\n",
                    "print(f\"Unified Classes: {cfg['unified_classes']}\")\n",
                    "print(f\"Crop Classes: {cfg['crop_classes']}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Dataset Manifest & Source Breakdown\n",
                    "We read the unified split manifest generated during Phase 1 data preparation."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "manifest_path = 'processed/split_manifest.csv'\n",
                    "if os.path.exists(manifest_path):\n",
                    "    df = pd.read_csv(manifest_path)\n",
                    "    print(f\"Total Images Processed: {len(df):,}\")\n",
                    "    print(\"\\nImages per Source Dataset:\")\n",
                    "    print(df['source_dataset'].value_counts())\n",
                    "    print(\"\\nImages per Split:\")\n",
                    "    print(df['split'].value_counts())\n",
                    "else:\n",
                    "    print(\"Manifest not yet generated. Run data preparation scripts.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Deduplication Summary\n",
                    "Perceptual hashing (`imagehash.dhash` 64-bit) was computed across all raw images to detect and prune near-identical video frame sequences."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "if 'is_duplicate' in df.columns:\n",
                    "    n_dup = (df['is_duplicate'] == True).sum()\n",
                    "    n_unique = (df['is_duplicate'] == False).sum()\n",
                    "    print(f\"Unique Real Scenes: {n_unique:,} ({n_unique / len(df) * 100:.1f}%)\")\n",
                    "    print(f\"Duplicates / Video redundancy pruned: {n_dup:,} ({n_dup / len(df) * 100:.1f}%)\")\n",
                    "    \n",
                    "    # Plot duplicate distribution\n",
                    "    plt.figure(figsize=(6, 4))\n",
                    "    plt.bar(['Unique Scenes', 'Pruned Duplicates'], [n_unique, n_dup], color=['#2b5c8f', '#c94a29'])\n",
                    "    plt.ylabel('Image Count')\n",
                    "    plt.title('Perceptual Hash Deduplication')\n",
                    "    plt.grid(axis='y', linestyle='--', alpha=0.5)\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Generative Worker Crop Statistics (64x64)\n",
                    "Loading compressed `.npz` crop files for training and evaluation."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "crops_dir = 'processed/crops'\n",
                    "split_data = {}\n",
                    "class_labels = {0: 'Compliant', 1: 'No Helmet', 2: 'No Vest', 3: 'Both Missing'}\n",
                    "\n",
                    "for s in ['train', 'val', 'test', 'final_test']:\n",
                    "    p = os.path.join(crops_dir, f'crops_{s}_64.npz')\n",
                    "    if os.path.exists(p):\n",
                    "        d = np.load(p, allow_pickle=True)\n",
                    "        split_data[s] = d\n",
                    "        counts = pd.Series(d['labels']).value_counts().sort_index()\n",
                    "        print(f\"=== Split: {s.upper()} ({len(d['labels']):,} crops) ===\")\n",
                    "        for c_id, name in class_labels.items():\n",
                    "            cnt = counts.get(c_id, 0)\n",
                    "            pct = cnt / len(d['labels']) * 100 if len(d['labels']) > 0 else 0\n",
                    "            print(f\"  Class {c_id} ({name:<12}): {cnt:>6,} ({pct:>5.1f}%)\")\n",
                    "        print()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Class Imbalance Visualization\n",
                    "Plotting the class distribution in the training split."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "if 'train' in split_data:\n",
                    "    tr_labels = split_data['train']['labels']\n",
                    "    counts = [int((tr_labels == i).sum()) for i in range(4)]\n",
                    "    names = [class_labels[i] for i in range(4)]\n",
                    "    colors = ['#2ca02c', '#d62728', '#ff7f0e', '#9467bd']\n",
                    "    \n",
                    "    plt.figure(figsize=(8, 4.5))\n",
                    "    bars = plt.bar(names, counts, color=colors)\n",
                    "    plt.ylabel('Number of 64x64 Worker Crops')\n",
                    "    plt.title('Training Split Worker Crop Distribution & Severe Imbalance')\n",
                    "    plt.grid(axis='y', linestyle='--', alpha=0.5)\n",
                    "    \n",
                    "    for bar in bars:\n",
                    "        yval = bar.get_height()\n",
                    "        plt.text(bar.get_x() + bar.get_width()/2.0, yval + max(counts)*0.015, f\"{int(yval):,}\", ha='center', va='bottom', fontweight='bold')\n",
                    "        \n",
                    "    plt.tight_layout()\n",
                    "    plt.savefig('results/figures/class_imbalance.png', dpi=300)\n",
                    "    plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Sample Inspection Grids\n",
                    "Displaying sample grids for visual verification across compliant workers and violation categories."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "grid_files = [\n",
                    "    ('Compliant (Class 0)', 'outputs/sample_grids/grid_class_0_compliant.png'),\n",
                    "    ('No Helmet (Class 1)', 'outputs/sample_grids/grid_class_1_no_helmet.png'),\n",
                    "    ('No Vest (Class 2)', 'outputs/sample_grids/grid_class_2_no_vest.png'),\n",
                    "    ('Both Missing (Class 3)', 'outputs/sample_grids/grid_class_3_both_missing.png'),\n",
                    "    ('Overview (All 4 Classes)', 'outputs/sample_grids/overview_all_classes_grid.png')\n",
                    "]\n",
                    "\n",
                    "for title, path in grid_files:\n",
                    "    if os.path.exists(path):\n",
                    "        print(f\"=== {title} ===\")\n",
                    "        im = Image.open(path)\n",
                    "        plt.figure(figsize=(7, 7))\n",
                    "        plt.imshow(im)\n",
                    "        plt.axis('off')\n",
                    "        plt.title(title, fontsize=14, fontweight='bold')\n",
                    "        plt.tight_layout()\n",
                    "        plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Conclusions & Ready for Phase 2\n",
                    "- Preprocessed crop arrays (`.npz`) are stored in RAM / memory-mapped ready format.\n",
                    "- Final test set (`processed/final_test/`) is completely isolated and will not be touched by any generative training.\n",
                    "- Verified that resolution 64x64 is ready for 100-epoch training across the 5 generative models."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    nb_path = "notebooks/00_data_overview.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {nb_path}")


if __name__ == "__main__":
    create_overview_notebook()
