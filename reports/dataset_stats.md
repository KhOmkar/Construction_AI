# Dataset Statistics & Imbalance Report

**Date:** 2026-11-01  
**Project:** Construction Site AI Safety Monitoring  
**Target Resolution:** 64x64 uint8 (Tier B RTX 3050 Optimized)  

---

## 1. Executive Summary

Phase 1 data preparation has successfully unified, cleaned, deduplicated, and extracted worker crops across all approved real-world construction datasets.

- **Total Source Images Processed:** 25472
- **Duplicates Pruned:** 6927 images
- **Effective Real Scenes:** 18545
- **Total Worker Crops Extracted (64x64):** 33047

---

## 2. Generative Model Worker Crop Counts by Split

| Split | Compliant (0) | No Helmet (1) | No Vest (2) | Both Missing (3) | Total Crops |
|---|---|---|---|---|---|
| **train** | 13,297 | 1,326 | 3,735 | 1,410 | **19,768** |
| **val** | 1,644 | 166 | 433 | 153 | **2,396** |
| **test** | 1,743 | 174 | 440 | 167 | **2,524** |
| **final_test** | 7,409 | 950 | 0 | 0 | **8,359** |
| **ALL COMBINED** | **24,093** | **2,616** | **4,608** | **1,730** | **33,047** |

---

## 3. Class Imbalance Analysis (Training Split)

In real construction environments, compliant workers far outnumber safety violations. This natural imbalance is preserved in the dataset and establishes the direct motivation for generative modeling (augmenting rare violations with GAN, Diffusion, and Transformer):

- **Compliant (Class 0):** 13,297 crops (67.3%)
- **Missing Helmet (Class 1):** 1,326 crops (6.7%) — **Imbalance Ratio 1 : 10.0**
- **Missing Safety Vest (Class 2):** 3,735 crops (18.9%) — **Imbalance Ratio 1 : 3.6**
- **Both Missing (Class 3):** 1,410 crops (7.1%) — **Imbalance Ratio 1 : 9.4**

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
