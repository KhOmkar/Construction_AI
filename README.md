# 🦺 Construction Site AI: Generative Safety Monitoring

An industrial computer vision and deep generative learning system for workplace safety compliance inspection on active construction sites. The system investigates five deep generative paradigms for monitoring personal protective equipment (PPE) compliance—specifically hard-hat and high-visibility vest presence—optimized under strict compute and memory constraints.

---

## 📌 Project Overview

Construction environments present severe occupational hazards where adherence to PPE regulations directly correlates with injury prevention. This project explores five generative architectures to address critical challenges in automated visual inspection:
- **Class Imbalance & Rare Violations:** Real-world compliance rates exceed 95%, causing catastrophic deficit in safety violation data.
- **Continuous Latent Representation:** Mapping worker appearances onto continuous manifolds for safety-state interpolation.
- **Semi-Supervised Anomaly Detection:** Flagging unseen non-compliant gear without requiring explicit violation labels during training.
- **Photorealistic Violation Synthesis:** Generating high-fidelity synthetic non-compliant worker crops to augment downstream object detectors.

### 🏷️ Unified Safety Ontology

All disparate annotations are harmonized into a unified safety schema:
| Class ID | Label | Description |
| :---: | :--- | :--- |
| **0** | `compliant` | Fully compliant worker wearing both a hard-hat and high-visibility vest |
| **1** | `no_helmet` | Safety violation: worker head without a hard-hat / safety helmet |
| **2** | `no_vest` | Safety violation: worker torso without a high-visibility reflective vest |
| **3** | `both_missing` | Critical violation: worker missing both hard-hat and safety vest |

---

## 🏛️ Generative Architectures Roster

| # | Architecture | Paradigm & Safety Role | Key Mechanism / Policy |
| :---: | :--- | :--- | :--- |
| **1** | **Convolutional Autoencoder** | Semi-Supervised Anomaly Detection | Trained exclusively on compliant workers ($y=0$); reconstruction error heatmaps flag missing safety equipment. |
| **2** | **Conditional VAE (C-VAE)** | Smooth Generative Latent Manifold | Linear KL warmup ($\beta: 0 \to 1.0$) over 10 epochs; continuous latent interpolation across compliance states. |
| **3** | **Conditional GAN (DCGAN)** | Adversarial Rare Violation Synthesis | Spectral Normalization with DiffAugment (color, translation, cutout) for rapid generation without mode collapse. |
| **4** | **Conditional DDPM / DDIM** | High-Fidelity Diffusion Synthesis | Diffusers UNet2D with EMA ($0.9999$), 10% label dropout, and Classifier-Free Guidance ($w=3.0$). |
| **5** | **Two-Stage VQ-VAE + ImageGPT** | Discrete Autoregressive Token Prior | Stage 1 learns a 512-entry codebook; Stage 2 trains an autoregressive causal transformer over discrete tokens. |

---

## 🖥️ Streamlit Interactive Safety Dashboard

The repository features an interactive web dashboard built with Streamlit for live inspection, demonstration, and evaluation.

```
dashboard/
├── app.py                            # Main entrypoint with GPU monitor & architectural roster
└── pages/
    ├── 1_Data_Explorer.py            # Dataset split inspection, safety schemas & crop galleries
    ├── 2_Model_Arena.py              # Interactive crop synthesizer (VAE, GAN, Diffusion, Transformer)
    ├── 3_Anomaly_Detector.py         # Reconstruction error heatmaps with calibrated τ* threshold
    ├── 4_Comparative_Benchmark.py     # Cross-model evaluation leaderboards & comparison grids
    └── 5_About.py                    # Project ethics, privacy safeguards, and licensing
```

### Running the Dashboard Locally

1. **Activate your environment** and install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch the Streamlit dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```
   Or use the provided shortcut scripts:
   - **Windows:** `scripts\run_dashboard.bat`
   - **Linux / macOS:** `bash scripts/run_dashboard.sh`

3. Open your browser at `http://localhost:8501`.

---

## 🐳 Docker & Container Deployment

For team members deploying the dashboard to cloud environments, staging servers, or container services (e.g., AWS ECS, GCP Cloud Run, Render, Hugging Face Spaces):

### Using Docker Compose (Recommended)

```bash
docker compose up --build
```
The application will be accessible at `http://localhost:8501`.

### Using the Docker CLI Directly

```bash
# Build the image
docker build -t construction-ai-dashboard:latest .

# Run the container
docker run -d -p 8501:8501 --name construction_ai construction-ai-dashboard:latest
```

### Deployment Dependencies

A dedicated `requirements-dashboard.txt` file is provided with lightweight CPU-compatible PyTorch wheels and dependencies for fast build times and reduced container footprint (< 1.5 GB).

---

## 📂 Repository Structure

```
├── configs/                          # YAML hyperparameters for all 5 generative architectures
│   ├── ae.yaml
│   ├── dataset.yaml
│   ├── diffusion.yaml
│   ├── gan.yaml
│   ├── vae.yaml
│   └── vqvae_transformer.yaml
├── dashboard/                        # Multi-page interactive Streamlit dashboard
│   ├── app.py
│   └── pages/
├── notebooks/                        # Jupyter analysis and model exploration notebooks
│   ├── 00_data_overview.ipynb
│   ├── 01_autoencoder.ipynb
│   ├── 02_vae.ipynb
│   ├── 03_gan.ipynb
│   ├── 04_diffusion.ipynb
│   ├── 05_transformer.ipynb
│   └── 06_model_comparison.ipynb
├── outputs/                          # Reference evaluation grids and interpolation visuals
│   ├── classifier/
│   ├── cvae_eval/
│   └── sample_grids/
├── reports/                          # Comprehensive technical audits and specifications
│   ├── data_audit.md
│   ├── dataset_stats.md
│   ├── final_report.md
│   ├── hardware.md
│   ├── licenses.md
│   └── metrics_definitions.md
├── results/                          # Benchmark evaluation leaderboards & comparative figures
│   ├── figures/
│   │   └── cross_model_comparison_grid.png
│   ├── tables/
│   │   ├── benchmark_table.md
│   │   └── benchmark_table.tex
│   └── all_runs.csv
├── runs/
│   └── ae_baseline/
│       └── anomaly_eval.json         # Calibrated AUROC & anomaly threshold (τ*) for dashboard
├── scripts/                          # One-click dashboard execution scripts
│   ├── run_dashboard.bat
│   └── run_dashboard.sh
├── src/                              # Modular source code
│   ├── common/                       # Metrics, dataset loaders, diffaugment, visualization
│   ├── data_prep/                    # Conversion, deduplication, crop extraction, pipeline
│   ├── models/                       # Model definitions (AE, VAE, GAN, Diffusion, VQ-VAE, ImageGPT)
│   ├── eval_benchmark.py             # Comprehensive multi-model evaluation harness
│   ├── generate_benchmark_visuals.py # Side-by-side comparative grid generator
│   ├── run_remaining_pipeline.py     # Orchestration pipeline runner
│   ├── train.py                      # Multi-architecture training suite
│   └── train_classifier.py           # Reference Oracle classifier training
├── .dockerignore
├── .gitignore
├── .streamlit/config.toml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt                  # Full local training dependencies
├── requirements-dashboard.txt        # Lightweight dashboard deployment dependencies
└── README.md
```

---

## 📊 Benchmark Leaderboard

Quantitative evaluation across 2,048 samples per model against the quarantined real test split ($N = 2,524$):

| Architecture | Paradigm | FID ↓ | KID ↓ | IS ↑ | CCA Acc % ↑ | Diversity (LPIPS) ↑ | Novelty (L2) ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Convolutional AE** | Anomaly Detection | 158.42 | 0.141 | 1.82 | 44.2% | 0.185 | 0.082 |
| **Conditional VAE** | Latent Interpolation | 134.18 | 0.118 | 2.15 | 68.4% | 0.241 | 0.104 |
| **Conditional GAN** | Adversarial Synthesis | 82.50 | 0.071 | 2.68 | 71.9% | 0.312 | 0.138 |
| **DDPM / DDIM** | Diffusion with CFG | **54.20** | **0.038** | **3.14** | **81.5%** | **0.378** | **0.176** |
| **VQ-VAE + ImageGPT** | Discrete Token Prior | 94.60 | 0.089 | 2.45 | 65.2% | 0.289 | 0.125 |

---

## 👥 Notes for Collaborators (Dashboard & Deployment)

1. **Dashboard Modifications:**
   - Add new analytical views or pages inside `dashboard/pages/`. Streamlit auto-discovers numbered scripts.
   - For styling, tweak `.streamlit/config.toml`.
2. **Model Inference in Dashboard:**
   - `dashboard/pages/2_Model_Arena.py` and `dashboard/pages/3_Anomaly_Detector.py` dynamically check for weights in `runs/`. If weights are missing, the UI gracefully prompts the user with fallback guidance or file upload options.
3. **Deployment Target Considerations:**
   - If deploying to cloud instances with limited memory, use `requirements-dashboard.txt` to avoid pulling unnecessary CUDA overhead.
   - Ports can be remapped in `docker-compose.yml` or passed via `STREAMLIT_SERVER_PORT`.

---

## 📜 Ethics, Privacy & Licensing

- **Face De-Identification:** Visual crops focus strictly on safety apparel (torso and hardhat boundary), omitting high-resolution facial biometrics.
- **Worker Privacy:** This system inspects PPE equipment compliance for occupational health and safety hazard mitigation, not for worker surveillance or productivity tracking.
- **Licenses:** Raw sources audited under CC-BY 4.0, MIT, and academic research licenses. Copyleft (AGPL) packages were strictly excluded to ensure academic and commercial compatibility.
