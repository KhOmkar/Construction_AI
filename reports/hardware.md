# Hardware Environment & Configuration Report

**Date:** 2026-11-01  
**Target Tier:** **Tier B** (NVIDIA GPU with 6 GB VRAM)  
**Execution Environment:** Local PC (`d:\Construction_PEC`)

---

## 1. System Specifications

| Component | Detected Specification | Target / Expected | Status |
|---|---|---|---|
| **Operating System** | Windows 11 (Build 10.0.26200, AMD64) | Windows 11 | **Match** |
| **CPU** | Intel64 Family 6 Model 183 (10 Physical / 16 Logical cores) | Modern Multi-core CPU | **Pass** |
| **System Memory (RAM)** | 15.71 GB Total (~3.8 GB Available) | ≥ 16 GB | **Pass** |
| **Project Storage (Drive D:)** | **183.03 GB Free** (301.59 GB Total) | > 50 GB | **Optimal** |
| **System Storage (Drive C:)** | 20.46 GB Free (172.94 GB Total) | > 10 GB | **Pass** |
| **GPU Model** | **NVIDIA GeForce RTX 3050 6GB Laptop GPU** | RTX 3050 6GB | **Exact Match** |
| **Dedicated VRAM** | **6,144 MB (6.00 GB)** | 6.0 GB | **Tier B** |
| **GPU Architecture** | Ampere (Compute Capability 8.6) | Ampere (SM 8.6) | **Pass** |
| **Driver / CUDA Driver** | Driver 610.88 / CUDA 13.3 | ≥ 525.x / ≥ 12.x | **Pass** |
| **GPU Temperature / Power** | 49°C / 11W idle (95W TGP Max) | Operating nominal | **Pass** |
| **Python Version** | **Python 3.12.10** | Python 3.10 - 3.12 | **Pass** |
| **PyTorch Version** | **PyTorch 2.6.0+cu124** | PyTorch with CUDA | **Installed & Active** |
| **CUDA Runtime / cuDNN** | CUDA 12.4 / cuDNN 9.0.1 | CUDA 12.x | **Match** |

---

## 2. Tier B Operating Constraints & Strategy

Per Section 2 and Section 7 of the Local Implementation Plan:
1. **Resolution:** Fixed to **64x64**. (128x128 is not planned for 6 GB Tier B to prevent out-of-memory errors).
2. **Precision:** Automatic Mixed Precision (`torch.amp.autocast('cuda')` with `GradScaler` in fp16).
3. **Batch Sizes:**
   - Autoencoder: 128
   - VAE / C-VAE: 128
   - Conditional GAN: 64
   - Conditional Diffusion: 32 (with gradient accumulation = 4 for effective batch size 128)
   - VQ-VAE: 128
   - Transformer: 32 - 64
4. **Data Loading:**
   - Preprocessed crops packaged into compressed `.npz` arrays and loaded directly into RAM or memory-mapped (`np.load(..., mmap_mode='r')`).
   - `num_workers=0` (or `2` carefully tested on Windows) to eliminate multiprocessing overhead and IPC memory bottlenecks.
5. **Training Length:**
   - Exactly **100 epochs** per generative model.
   - Resumable from latest checkpoint automatically if interrupted.
6. **Detector Experiment Override:**
   - The original Colab YOLO detector experiment is explicitly omitted; the focus is entirely on the five generative models, their evaluation, and the local Streamlit dashboard.
