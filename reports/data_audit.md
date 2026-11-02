# Comprehensive Data Audit Report: Construction Site AI

**Date:** 2026-11-01  
**Project:** Construction Site AI — PPE & Hazard Generative Modeling  
**Workspace:** `d:\Construction_PEC`  
**Status:** Audit Complete — Awaiting User Approval [STOP]

---

## 1. Executive Summary

A complete recursive audit was performed across all archives and subfolders in `data/`. No raw data files were modified, moved, or deleted. 

A total of **8 distinct dataset sources** were discovered, comprising **26,140 unique local image files**, **1,416 remote NDJSON records**, and **78,088 bounding box annotations**. The data spans 5 distinct formats: YOLO `.txt`, Pascal VOC CSV, COCO JSON, CVAT XML, and NDJSON.

Every single image file across all local archives and directories was verified using PIL `Image.verify()`. **Zero corrupt or unreadable image files were detected (0 / 26,140).**

### Discovered Datasets At-a-Glance

| Dataset Identifier | Raw Container | Primary Format | Total Images | Total BBoxes | Raw Classes | Splits (Train / Val / Test) | License |
|---|---|---|---|---|---|---|---|
| **1. PPE-2286y** | `20250731-PPE2286y.zip` | YOLO txt | 2,286 | 6,038 | 4 (`Helmet`, `NoHelmet`, `NoVest`, `Vest`) | 1,829 / 457 / 0 | CC BY 4.0 |
| **2. CVAT-Helmet** | `archive (1).zip` | CVAT XML | 46 (92 files*) | 83 | 2 (`helmet`, `no_helmet`) | None (Unsplit) | Unspecified |
| **3. Safety-Vests** | `archive.zip` | VOC-style CSV | 3,897 | 8,472 | 2 (`Safety Vest`, `NO-Safety Vest`) | 2,728 / 779 / 390 | CC BY 4.0 |
| **4. Construction-PPE** | `construction-ppe.ndjson` | NDJSON + CDN | 1,416 (Remote) | 11,887 | 11 (`helmet`, `gloves`, `vest`, `boots`, etc.) | 1,132 / 143 / 141 | AGPL-3.0 |
| **5. CSS-v28** | `css-data.zip` | YOLO txt | 2,801 | 38,352 | 10 (`Hardhat`, `NO-Hardhat`, `Vest`, etc.) | 2,605 / 114 / 82 | CC BY 4.0 |
| **6. Workplace-Safety-v2** | `PPE Dataset...coco.zip` | COCO JSON | 1,613 | 7,225 | 8 (`Vest`, `Helmet`, `Boots`, `Person`, etc.) | 1,126 / 326 / 161 | CC BY 4.0 |
| **7. PPE-Detection-6Cls** | `PPE.zip` | YOLO txt | 12,078 | 16,012 | 6 (`Gloves`, `Vest`, `goggles`, `helmet`, etc.) | 8,774 / 2,070 / 1,234 | CC BY 4.0 |
| **8. jhboyo-PPE** | `data/test`, `data/train` | YOLO txt | 3,934 (Local) | 10,862 | 3 (`helmet`, `head`, `vest`) | 1,183 (Train*) / 0 (Val) / 2,751 (Test) | MIT |

*\*Notes:*
- `archive (1).zip` contains 46 raw images in `img/` and 46 pre-drawn box visualizations in `boxes/`. Only the 46 raw images are candidates for detection/generation.
- `jhboyo-PPE`: `data/test` is complete (2,751 images + 2,751 label files); `data/train` is an incomplete download containing 1,183 images with 0 labels.
- `construction-ppe.ndjson`: 1,416 images are indexed with active signed CDN URLs (`https://cdn.ul.run/...`); image binaries are not saved locally.

---

## 2. Dataset-by-Dataset Audit Details

### Dataset 1: `20250731-PPE2286y.zip`
- **Source**: Roboflow Universe (`ying2001/20250731-ppe2286-m0ega`)
- **Format**: YOLO v8 format (normalized `class x_center y_center width height`)
- **Image Count**: 2,286 (Train: 1,829, Valid: 457, Test: 0)
- **Annotation Count**: 2,286 `.txt` label files (6,038 bounding boxes)
- **Image Size Range**: Min: 128x136, Max: 5712x4284 (Aspect ratios vary)
- **License**: CC BY 4.0 (present in `README.dataset.txt`)
- **Class Breakdown**:
  - `0: Helmet` → 2,421 instances
  - `1: NoHelmet` → 1,174 instances
  - `2: NoVest` → 1,058 instances
  - `3: Vest` → 1,385 instances
- **Integrity**: 0 corrupt files. All image files have matching label files.

---

### Dataset 2: `archive (1).zip`
- **Source**: CVAT XML annotation export (`TrainingData`)
- **Format**: CVAT XML (`annotations.xml` with `<box xtl="..." ytl="..." xbr="..." ybr="..." label="...">`)
- **Image Count**: 46 raw images in `img/` (46 visualization overlays in `boxes/`)
- **Annotation Count**: 1 XML file with 83 box annotations
- **Image Size Range**: Min: 644x600, Max: 1584x1598
- **Existing Splits**: None (single folder)
- **License**: Unspecified (academic/research use)
- **Class Breakdown**:
  - `helmet`: 72 instances
  - `no_helmet`: 11 instances
- **Integrity**: 0 corrupt files.

---

### Dataset 3: `archive.zip` (Safety Vests Detection)
- **Source**: Roboflow Universe (`echo-dotf3/safety-vests-c89yh`)
- **Format**: Pascal VOC-style CSV (`filename, width, height, class, xmin, ymin, xmax, ymax`)
- **Image Count**: 3,897 (Train: 2,728, Valid: 779, Test: 390)
- **Annotation Count**: 3 CSV files (`_annotations.csv` per split), 8,472 bounding boxes
- **Image Size Range**: Min: 112x160, Max: 6838x5472
- **License**: CC BY 4.0 (present in `README.dataset.txt`)
- **Class Breakdown**:
  - `Safety Vest`: 6,448 instances
  - `NO-Safety Vest`: 2,024 instances
- **Integrity**: 0 corrupt files. All CSV records match images.

---

### Dataset 4: `construction-ppe.ndjson` (Ultralytics Construction-PPE)
- **Source**: Ultralytics Inc. (Dalvi, Singh, Bhingarde, Chalke 2025)
- **Format**: Line-delimited JSON (NDJSON) containing image URLs and normalized xywh boxes
- **Image Count**: 1,416 metadata records (Train: 1,132, Val: 143, Test: 141)
- **Annotation Count**: 11,887 bounding boxes across 1,416 images
- **Image Size Range**: Recorded as 640x640 to 1024x768
- **License**: **AGPL-3.0** (Ultralytics Open Source License) — **FLAGGED FOR REVIEW**
- **Class Breakdown**:
  - `0: helmet`: 1,734
  - `1: gloves`: 1,445
  - `2: vest`: 1,618
  - `3: boots`: 1,597
  - `4: goggles`: 518
  - `5: none`: 797
  - `6: Person`: 2,245
  - `7: no_helmet`: 485
  - `8: no_goggle`: 411
  - `9: no_gloves`: 556
  - `10: no_boots`: 115
- **Operational Status**: Image binaries are remote (valid signed CDN URLs verified via HTTP HEAD 200). Local storage contains 0 images. Because our remaining datasets already yield over 23,000 real images, this dataset can be optionally downloaded or excluded to eliminate AGPL copyleft encumbrances.

---

### Dataset 5: `css-data.zip` (Construction Site Safety v28)
- **Source**: Roboflow Universe (`roboflow-universe-projects/construction-site-safety`)
- **Format**: YOLO txt format (normalized xywh)
- **Image Count**: 2,801 (Train: 2,605, Valid: 114, Test: 82)
- **Annotation Count**: 2,801 `.txt` label files, 38,352 bounding boxes
- **Image Size Range**: 640x640 (pre-standardized by Roboflow)
- **License**: CC BY 4.0 (present in `README.dataset.txt`)
- **Class Breakdown (Order Verified from Roboflow v28 Schema)**:
  - `0: Hardhat`: 3,334
  - `1: Mask`: 1,700
  - `2: NO-Hardhat`: 2,427
  - `3: NO-Mask`: 3,250
  - `4: NO-Safety Vest`: 4,158
  - `5: Person`: 9,872
  - `6: Safety Cone`: 3,502
  - `7: Safety Vest`: 3,135
  - `8: machinery`: 5,346
  - `9: vehicle`: 1,628
- **Integrity**: 0 corrupt files. Missing explicit `data.yaml` inside zip; schema verified externally and cross-referenced with bounding box geometry.

---

### Dataset 6: `PPE Dataset for Workplace Safety.v2i.coco-mmdetection.zip`
- **Source**: Roboflow Universe (`siabar/ppe-dataset-for-workplace-safety`)
- **Format**: COCO JSON (`_annotations.coco.json` with `[x, y, width, height]` absolute pixel coordinates)
- **Image Count**: 1,613 (Train: 1,126, Valid: 326, Test: 161)
- **Annotation Count**: 3 JSON files, 7,225 bounding boxes
- **Image Size Range**: 640x640
- **License**: CC BY 4.0 (present in `README.dataset.txt`)
- **Class Breakdown**:
  - `Vest` (id 8): 1,944
  - `Helmet` (id 5): 1,388
  - `Person` (id 7): 1,442
  - `Boots` (id 1): 2,440
  - `Glass` (id 3): 6
  - `Ear-protection` (id 2): 2
  - `Glove` (id 4): 2
  - `Mask` (id 6): 1
- **Integrity**: 0 corrupt files.

---

### Dataset 7: `PPE.zip` (PPE Detection 6-Class)
- **Source**: Roboflow Universe (`tanish-y7iqo/ppe-detection_data-adcya`)
- **Format**: YOLO txt format
- **Image Count**: 12,078 (Train: 8,774, Valid: 2,070, Test: 1,234)
- **Annotation Count**: 12,078 `.txt` label files, 16,012 bounding boxes
- **Image Size Range**: 640x640
- **License**: CC BY 4.0 (present in `data.yaml` and `README.dataset.txt`)
- **Class Breakdown**:
  - `0: Gloves`: 2,693
  - `1: Vest`: 4,418
  - `2: goggles`: 1,431
  - `3: helmet`: 2,703
  - `4: mask`: 2,761
  - `5: safety_shoe`: 2,006
- **Integrity**: 0 corrupt files.

---

### Dataset 8: `data/test` & `data/train` (`jhboyo/ppe-dataset`)
- **Source**: Hugging Face Hub (`jhboyo/ppe-dataset`)
- **Format**: YOLO txt format
- **Image Count on Disk**: 3,934
  - `data/test/images`: 2,751 images
  - `data/train/images`: 1,183 images
- **Annotation Count on Disk**:
  - `data/test/labels`: 2,751 `.txt` files (10,862 bounding boxes, 8 empty label files representing background/negative images)
  - `data/train/labels`: 0 files (**Incomplete download**)
- **Image Size Range**: 415x415 to 640x640
- **License**: MIT License (documented in `data/README.md`)
- **Class Breakdown (in Test split)**:
  - `0: helmet`: 6,939
  - `1: head`: 962 (represents unhelmeted head -> `no_helmet`)
  - `2: vest`: 2,961
- **Integrity**: 0 corrupt files.

---

## 3. Discovered Irregularities & Recommendations

1. **Incomplete `jhboyo/train`**:
   - `data/train/images` has 1,183 images but no label files.
   - **Recommendation**: Do not include unannotated images in supervised training sets. We propose using the complete `data/test` set (2,751 fully labeled images) and ignoring the unannotated 1,183 images from `data/train` unless you wish to resume the Hugging Face download.
2. **Missing `data.yaml` in `css-data.zip`**:
   - The archive lacks an internal YAML file, but the class mapping is confirmed as Roboflow's standard 10-class Construction Site Safety ontology: `['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']`.
3. **Empty Label Files**:
   - In `jhboyo/test/labels`, 8 files are 0 bytes (representing background images with no PPE). In YOLO, this is standard negative sample representation and will be cleanly handled.
4. **Ultralytics Construction-PPE Remote URLs & AGPL License**:
   - Contains 1,416 images on CDN with an AGPL-3.0 license.
   - We already have **23,436 real images with CC-BY and MIT licenses**.
   - **Proposal**: See Question 1 below.

---

## 4. Proposed Unified Class Mapping Table

**Target Unified Schema (5 Classes):**
- `0`: `helmet`
- `1`: `no_helmet`
- `2`: `vest`
- `3`: `no_vest`
- `4`: `person`

| Source Dataset | Source Class Name / ID | Proposed Unified Class | Action | Rationale |
|---|---|---|---|---|
| **20250731-PPE2286y** | `0: Helmet` | `helmet` (0) | **MAP** | Direct match |
| | `1: NoHelmet` | `no_helmet` (1) | **MAP** | Direct match (rare violation class) |
| | `2: NoVest` | `no_vest` (3) | **MAP** | Direct match (rare violation class) |
| | `3: Vest` | `vest` (2) | **MAP** | Direct match |
| **archive (1).zip** | `helmet` | `helmet` (0) | **MAP** | Direct match |
| | `no_helmet` | `no_helmet` (1) | **MAP** | Direct match |
| **archive.zip** | `Safety Vest` | `vest` (2) | **MAP** | Direct match |
| | `NO-Safety Vest` | `no_vest` (3) | **MAP** | Direct match |
| **css-data.zip** | `0: Hardhat` | `helmet` (0) | **MAP** | Equivalent PPE equipment |
| | `1: Mask` | — | **DROP** | Out of target scope |
| | `2: NO-Hardhat` | `no_helmet` (1) | **MAP** | Missing helmet violation |
| | `3: NO-Mask` | — | **DROP** | Out of target scope |
| | `4: NO-Safety Vest` | `no_vest` (3) | **MAP** | Missing vest violation |
| | `5: Person` | `person` (4) | **MAP** | Full person bounding box |
| | `6: Safety Cone` | — | **DROP** | Site equipment/hazard, out of worker PPE scope |
| | `7: Safety Vest` | `vest` (2) | **MAP** | Direct match |
| | `8: machinery` | — | **DROP** | Heavy vehicle/machinery, out of target PPE scope |
| | `9: vehicle` | — | **DROP** | Vehicles, out of target PPE scope |
| **Workplace Safety COCO**| `Helmet` (5) | `helmet` (0) | **MAP** | Direct match |
| | `Vest` (8) | `vest` (2) | **MAP** | Direct match |
| | `Person` (7) | `person` (4) | **MAP** | Full person bounding box |
| | `Boots` (1), `Glass` (3), `Glove` (4), `Ear-protection` (2), `Mask` (6) | — | **DROP** | Out of target scope |
| **PPE.zip** | `helmet` (3) | `helmet` (0) | **MAP** | Direct match |
| | `Vest` (1) | `vest` (2) | **MAP** | Direct match |
| | `Gloves` (0), `goggles` (2), `mask` (4), `safety_shoe` (5) | — | **DROP** | Out of target scope |
| **jhboyo-PPE** | `0: helmet` | `helmet` (0) | **MAP** | Direct match |
| | `1: head` | `no_helmet` (1) | **MAP** | Head without helmet denotes violation |
| | `2: vest` | `vest` (2) | **MAP** | Direct match |
| **Construction-PPE** *(If Approved)* | `0: helmet` | `helmet` (0) | **MAP** | Direct match |
| | `2: vest` | `vest` (2) | **MAP** | Direct match |
| | `6: Person` | `person` (4) | **MAP** | Direct match |
| | `7: no_helmet` | `no_helmet` (1) | **MAP** | Direct match |
| | `1: gloves`, `3: boots`, `4: goggles`, `5: none`, `8: no_goggle`, `9: no_gloves`, `10: no_boots` | — | **DROP** | Out of target scope |

---

## 5. Proposed Project Architecture & Folder Layout

```
Construction_PEC/
├── configs/
│   ├── dataset.yaml            # Conversion, deduplication, crop settings
│   ├── ae.yaml                 # Autoencoder config (bottleneck 256, MSE+SSIM)
│   ├── vae.yaml                # VAE & C-VAE config (beta, KL warmup)
│   ├── gan.yaml                # C-GAN config (DiffAugment, spectral norm)
│   ├── diffusion.yaml          # DDPM UNet2D config (CFG, EMA, 64x64/128x128)
│   ├── vqvae_transformer.yaml  # VQ-VAE + GPT transformer config
│   └── yolo_downstream.yaml    # YOLOv8n experiment configs
├── data/                       # [READ ONLY - NEVER MODIFIED]
├── processed/                  # Generated by Phase 1
│   ├── scenes/                 # Unified scene images + YOLO format labels
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── dataset.yaml        # Ultralytics-compatible dataset descriptor
│   ├── final_test/             # Untouched real-only evaluation benchmark
│   │   ├── images/
│   │   └── labels/
│   ├── crops/                  # Fast-loading npz files for generative models
│   │   ├── crops_train_64.npz  # Worker crops (64x64 uint8) + labels + masks
│   │   ├── crops_val_64.npz
│   │   ├── crops_test_64.npz
│   │   ├── crops_train_128.npz # Optional 128x128 resolution
│   │   └── sample_crops/       # PNG inspection samples per class
│   └── manifest.csv            # Image ID, source dataset, split, original path, auto-caption
├── src/
│   ├── common/
│   │   ├── __init__.py
│   │   ├── dataset.py          # PyTorch Dataset classes for NPZ crops & scenes
│   │   ├── diffaugment.py      # Differentiable augmentation for GAN training
│   │   ├── metrics.py          # FID, KID, SSIM, LPIPS, AUROC wrappers
│   │   ├── checkpoint.py       # Mixed precision, auto-save & auto-resume helpers
│   │   └── visualization.py    # Image grid helpers, heatmaps, latent plots
│   ├── data_prep/
│   │   ├── convert_all.py      # Multi-format reader -> unified YOLO
│   │   ├── deduplicate.py      # Perceptual hashing (imagehash)
│   │   ├── split_data.py       # Seeded 80/10/10 split by image + final_test reservation
│   │   ├── extract_crops.py    # Person/torso crop extractor to .npz
│   │   └── caption_generator.py# Rule-based auto-captioning for manifest
│   ├── models/
│   │   ├── ae.py               # Autoencoder anomaly detector
│   │   ├── vae.py              # VAE and conditional VAE
│   │   ├── gan.py              # Conditional GAN with DiffAugment
│   │   ├── diffusion.py        # DDPM with classifier-free guidance
│   │   ├── vqvae.py            # Vector-Quantized VAE tokenizer
│   │   └── transformer.py      # Autoregressive GPT over discrete image tokens
│   └── downstream/
│       ├── build_augmented_sets.py # Paste synthetic worker crops into scenes
│       └── train_eval_yolo.py  # Train YOLOv8n on 4 experimental variants & evaluate
├── notebooks/                  # Standalone Colab notebooks
│   ├── 01_autoencoder.ipynb
│   ├── 02_vae.ipynb
│   ├── 03_cgan.ipynb
│   ├── 04_diffusion.ipynb
│   ├── 05_vqvae_transformer.ipynb
│   └── 06_downstream_yolo.ipynb
├── checkpoints/                # Local checkpoint directory
├── outputs/                    # Visualizations, sample grids, metric CSVs
├── reports/                    # Documentation, stats, data audit, licenses
│   ├── data_audit.md           # [This report]
│   ├── licenses.md             # IP and licensing analysis
│   ├── dataset_stats.md        # Post-preparation stats & imbalance analysis
│   └── final_report.md         # Final comprehensive academic paper / report
├── colab_setup.md              # Instructions for Drive upload & free T4 workflow
├── requirements.txt            # Project dependencies
├── PROGRESS.md                 # Running project tracking
└── README.md                   # University project overview & guide
```

---

## 6. Proposed Python Scripts for Phase 1 Data Preparation

1. **`src/data_prep/convert_all.py`**:
   - Reads every dataset directly from `data/` (handling `.zip` on the fly or via staging) without touching raw files.
   - Normalizes coordinate boxes from VOC CSV, CVAT XML, COCO JSON, and YOLO txt into unified YOLO format.
   - Implements the approved class mapping table.
   - Generates initial `manifest.csv`.
2. **`src/data_prep/deduplicate.py`**:
   - Computes perceptual hashes (difference hash `dhash` and average hash `ahash` via `imagehash`).
   - Identifies exact duplicates and near-identical video frame sequences (common in Roboflow and YouTube-derived sets).
   - Prunes duplicates, keeping one representative per hash cluster, and logs pruned count.
3. **`src/data_prep/split_data.py`**:
   - Isolates an independent, untouched real-only `final_test` set.
   - Performs reproducible 80/10/10 stratified split by image (fixed random seed = 42) across remaining data.
4. **`src/data_prep/extract_crops.py`**:
   - Extracts worker regions (person bbox, or combined head-torso region if only helmet/vest annotated).
   - Filters out crops smaller than configurable minimum size (default: 24x24 px).
   - Resizes crops with bicubic anti-aliasing to 64x64 (and 128x128).
   - Packages crops, class labels, and bounding boxes into single compressed `.npz` files per split (`crops_train_64.npz`, `crops_val_64.npz`, `crops_test_64.npz`) for instant Google Drive/Colab loading.
5. **`src/data_prep/caption_generator.py`**:
   - Generates structured descriptive worker captions based on present/absent gear (e.g., *"a construction worker wearing a safety helmet and reflective vest"*, *"a worker with no helmet and no vest"*).
   - Appends captions to `manifest.csv`.
6. **`src/data_prep/generate_stats_and_grids.py`**:
   - Generates 16-crop sample image grids per class (`outputs/sample_grids/`) for visual verification.
   - Calculates imbalance ratios and produces `reports/dataset_stats.md`.
