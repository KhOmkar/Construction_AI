"""
Generate comprehensive statistics in reports/dataset_stats.md
and 16-crop sample grids per class in outputs/sample_grids/ for visual label noise inspection.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def load_config(config_path="configs/dataset.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_image_grid(images, rows=4, cols=4, padding=4, bg_color=(240, 240, 240)):
    if len(images) == 0:
        return None
    h, w, c = images[0].shape
    grid_w = cols * w + (cols + 1) * padding
    grid_h = rows * h + (rows + 1) * padding
    grid_img = Image.new("RGB", (grid_w, grid_h), color=bg_color)

    for idx in range(min(len(images), rows * cols)):
        r = idx // cols
        c_idx = idx % cols
        x = padding + c_idx * (w + padding)
        y = padding + r * (h + padding)
        im = Image.fromarray(images[idx])
        grid_img.paste(im, (x, y))

    return grid_img


def generate_sample_grids(cfg):
    crops_dir = cfg["paths"]["crops_dir"]
    out_dir = cfg["paths"]["sample_grids_dir"]
    os.makedirs(out_dir, exist_ok=True)

    train_npz = os.path.join(crops_dir, "crops_train_64.npz")
    if not os.path.exists(train_npz):
        print(f"Error: {train_npz} not found.")
        return {}

    data = np.load(train_npz, allow_pickle=True)
    images = data["images"]
    labels = data["labels"]

    class_names = {
        0: "compliant",
        1: "no_helmet",
        2: "no_vest",
        3: "both_missing"
    }

    np.random.seed(cfg.get("seed", 42))
    grid_paths = {}

    overview_crops = []

    for cls_id, cls_name in class_names.items():
        indices = np.where(labels == cls_id)[0]
        if len(indices) == 0:
            print(f"Warning: No samples for class {cls_id} ({cls_name})")
            continue

        selected = np.random.choice(indices, size=min(16, len(indices)), replace=False)
        cls_images = images[selected]

        grid = create_image_grid(cls_images, rows=4, cols=4)
        if grid:
            grid_path = os.path.join(out_dir, f"grid_class_{cls_id}_{cls_name}.png")
            # Upscale 2x for sharp viewing in reports/artifacts
            grid_up = grid.resize((grid.width * 2, grid.height * 2), Image.Resampling.NEAREST)
            grid_up.save(grid_path)
            grid_paths[cls_name] = grid_path
            print(f"Saved {cls_name} grid to {grid_path}")

        # Pick 4 for combined overview
        sub_sel = selected[:4]
        for s in sub_sel:
            overview_crops.append(images[s])

    if len(overview_crops) == 16:
        overview_grid = create_image_grid(overview_crops, rows=4, cols=4)
        if overview_grid:
            ov_path = os.path.join(out_dir, "overview_all_classes_grid.png")
            ov_up = overview_grid.resize((overview_grid.width * 2, overview_grid.height * 2), Image.Resampling.NEAREST)
            ov_up.save(ov_path)
            grid_paths["overview"] = ov_path
            print(f"Saved overview grid to {ov_path}")

    return grid_paths


def compute_statistics_and_report(cfg, grid_paths):
    crops_dir = cfg["paths"]["crops_dir"]
    reports_dir = cfg["paths"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)

    split_manifest_path = os.path.join(cfg["paths"]["processed_dir"], "split_manifest.csv")
    df_manifest = pd.read_csv(split_manifest_path) if os.path.exists(split_manifest_path) else None

    crop_counts = {}
    class_names = {0: "compliant", 1: "no_helmet", 2: "no_vest", 3: "both_missing"}

    for split in ["train", "val", "test", "final_test"]:
        npz_path = os.path.join(crops_dir, f"crops_{split}_64.npz")
        if os.path.exists(npz_path):
            d = np.load(npz_path, allow_pickle=True)
            lbls = d["labels"]
            counts = {class_names[c]: int((lbls == c).sum()) for c in range(4)}
            counts["total"] = len(lbls)
            crop_counts[split] = counts

    # Imbalance calculation (train split)
    train_c = crop_counts.get("train", {})
    comp = train_c.get("compliant", 1)
    no_h = train_c.get("no_helmet", 1)
    no_v = train_c.get("no_vest", 1)
    both_m = train_c.get("both_missing", 1)

    r_h = comp / max(1, no_h)
    r_v = comp / max(1, no_v)
    r_both = comp / max(1, both_m)

    # Markdown report
    report_md = f"""# Dataset Statistics & Imbalance Report

**Date:** 2026-11-01  
**Project:** Construction Site AI Safety Monitoring  
**Target Resolution:** 64x64 uint8 (Tier B RTX 3050 Optimized)  

---

## 1. Executive Summary

Phase 1 data preparation has successfully unified, cleaned, deduplicated, and extracted worker crops across all approved real-world construction datasets.

- **Total Source Images Processed:** {len(df_manifest) if df_manifest is not None else 'N/A'}
- **Duplicates Pruned:** {(df_manifest['is_duplicate'] == True).sum() if df_manifest is not None and 'is_duplicate' in df_manifest else 'N/A'} images
- **Effective Real Scenes:** {(df_manifest['is_duplicate'] == False).sum() if df_manifest is not None and 'is_duplicate' in df_manifest else 'N/A'}
- **Total Worker Crops Extracted (64x64):** {sum(v.get('total', 0) for v in crop_counts.values())}

---

## 2. Generative Model Worker Crop Counts by Split

| Split | Compliant (0) | No Helmet (1) | No Vest (2) | Both Missing (3) | Total Crops |
|---|---|---|---|---|---|
"""
    for split in ["train", "val", "test", "final_test"]:
        c = crop_counts.get(split, {})
        report_md += f"| **{split}** | {c.get('compliant', 0):,} | {c.get('no_helmet', 0):,} | {c.get('no_vest', 0):,} | {c.get('both_missing', 0):,} | **{c.get('total', 0):,}** |\n"

    total_comp = sum(crop_counts.get(s, {}).get("compliant", 0) for s in crop_counts)
    total_no_h = sum(crop_counts.get(s, {}).get("no_helmet", 0) for s in crop_counts)
    total_no_v = sum(crop_counts.get(s, {}).get("no_vest", 0) for s in crop_counts)
    total_both = sum(crop_counts.get(s, {}).get("both_missing", 0) for s in crop_counts)
    total_all = sum(crop_counts.get(s, {}).get("total", 0) for s in crop_counts)

    report_md += f"| **ALL COMBINED** | **{total_comp:,}** | **{total_no_h:,}** | **{total_no_v:,}** | **{total_both:,}** | **{total_all:,}** |\n\n"

    report_md += f"""---

## 3. Class Imbalance Analysis (Training Split)

In real construction environments, compliant workers far outnumber safety violations. This natural imbalance is preserved in the dataset and establishes the direct motivation for generative modeling (augmenting rare violations with GAN, Diffusion, and Transformer):

- **Compliant (Class 0):** {comp:,} crops ({comp / max(1, train_c.get('total', 1)) * 100:.1f}%)
- **Missing Helmet (Class 1):** {no_h:,} crops ({no_h / max(1, train_c.get('total', 1)) * 100:.1f}%) — **Imbalance Ratio 1 : {r_h:.1f}**
- **Missing Safety Vest (Class 2):** {no_v:,} crops ({no_v / max(1, train_c.get('total', 1)) * 100:.1f}%) — **Imbalance Ratio 1 : {r_v:.1f}**
- **Both Missing (Class 3):** {both_m:,} crops ({both_m / max(1, train_c.get('total', 1)) * 100:.1f}%) — **Imbalance Ratio 1 : {r_both:.1f}**

---

## 4. Sample Verification Grids (16 Crops per Class)

The following PNG sample grids were generated in `outputs/sample_grids/` for visual quality and label noise inspection:

- **Compliant Grid (Class 0):** `outputs/sample_grids/grid_class_0_compliant.png`
- **No Helmet Grid (Class 1):** `outputs/sample_grids/grid_class_1_no_helmet.png`
- **No Vest Grid (Class 2):** `outputs/sample_grids/grid_class_2_no_vest.png`
- **Both Missing Grid (Class 3):** `outputs/sample_grids/grid_class_3_both_missing.png`
- **Overview Grid (All 4 Classes):** `outputs/sample_grids/overview_all_classes_grid.png`

> **Spot-Check Instructions [GATE 1]:**
> Inspect the sample grids to verify that compliant workers show helmets and high-visibility vests, and violation crops show exposed heads or ordinary clothing without vests.
"""

    report_path = os.path.join(reports_dir, "dataset_stats.md")
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write(report_md)

    print(f"\n[DONE] Statistical report generated: {report_path}")


def main():
    cfg = load_config()
    grid_paths = generate_sample_grids(cfg)
    compute_statistics_and_report(cfg, grid_paths)


if __name__ == "__main__":
    main()
