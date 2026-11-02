import os
import streamlit as st
import torch
import pandas as pd

st.set_page_config(
    page_title="Construction Site AI — Safety Monitoring",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar GPU status
st.sidebar.title("🦺 Construction Site AI")
st.sidebar.markdown("**Generative Workplace Safety System**")
st.sidebar.divider()

if torch.cuda.is_available():
    vram_alloc = torch.cuda.memory_allocated() / (1024**2)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    device_name = torch.cuda.get_device_name(0)
    st.sidebar.success(f"**GPU:** {device_name}")
    st.sidebar.metric("VRAM Usage", f"{vram_alloc:.1f} MB", f"Total: {vram_total:.0f} MB")
    st.sidebar.progress(min(1.0, vram_alloc / max(1.0, vram_total)))
else:
    st.sidebar.warning("Running on CPU mode")

st.sidebar.info("Navigate through the modules in the sidebar pages above.")

# Main page content
st.title("🦺 Construction Site AI: Generative Safety Monitoring")
st.markdown("""
Welcome to the **Construction Site AI** generative safety monitoring system. 
This university-level machine learning project investigates five generative deep learning architectures 
for visual worker safety inspection (hard-hat and high-visibility vest compliance) under a strict 
**NVIDIA RTX 3050 6GB Laptop GPU** hardware budget.
""")

st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Target Resolution", "64 × 64 px", "Unified uint8 crops")
col2.metric("Curated Worker Crops", "33,047", "Across 4 safety states")
col3.metric("Generative Models", "5 Architectures", "100 Epochs each")
col4.metric("Oracle Classifier Acc", "73.10%", "Reference safety benchmark")

st.divider()

st.subheader("Architectural Roster & Safety Roles")
roster_data = [
    {
        "Architecture": "1. Convolutional Autoencoder",
        "Role / Paradigm": "Semi-Supervised Anomaly Detection",
        "Training Policy": "Trained exclusively on compliant crops (y=0)",
        "Key Metric / Output": "Reconstruction error heatmaps & AUROC"
    },
    {
        "Architecture": "2. Conditional VAE (C-VAE)",
        "Role / Paradigm": "Smooth Generative Latent Manifold",
        "Training Policy": "Linear KL warmup (beta: 0 -> 1.0) over 10 epochs",
        "Key Metric / Output": "Continuous latent interpolation across states"
    },
    {
        "Architecture": "3. Conditional GAN (DCGAN)",
        "Role / Paradigm": "Adversarial Rare Violation Synthesis",
        "Training Policy": "Spectral Normalization + DiffAugment (color/trans/cutout)",
        "Key Metric / Output": "Instantaneous synthetic no_helmet / no_vest crops"
    },
    {
        "Architecture": "4. Class-Conditional DDPM/DDIM",
        "Role / Paradigm": "High-Fidelity Diffusion with CFG",
        "Training Policy": "diffusers UNet2D, EMA (0.9999), 10% label dropout",
        "Key Metric / Output": "50-step deterministic DDIM generation"
    },
    {
        "Architecture": "5. Two-Stage VQ-VAE + ImageGPT",
        "Role / Paradigm": "Discrete Autoregressive Token Prior",
        "Training Policy": "Stage 1: 512 codebook; Stage 2: 6-layer ImageGPT",
        "Key Metric / Output": "Autoregressive generation with temperature sweeps"
    }
]
st.dataframe(pd.DataFrame(roster_data), use_container_width=True, hide_index=True)

st.divider()

st.subheader("System Modules")
m1, m2 = st.columns(2)
with m1:
    st.markdown("""
    - **📂 [Data Explorer](Data_Explorer):** Inspect the raw dataset distribution, 5-class unified schema, and crop sample galleries.
    - **🎨 [Model Arena](Model_Arena):** Interactively synthesize photorealistic workers across 4 safety classes using VAE, GAN, Diffusion, or Transformer.
    """)
with m2:
    st.markdown("""
    - **🚨 [Anomaly Detector](Anomaly_Detector):** Test the Autoencoder anomaly detection system on worker crops with real-time error heatmaps.
    - **📊 [Comparative Benchmark](Comparative_Benchmark):** View cross-model evaluation leaderboards (FID, KID, IS, CCA, Diversity, Novelty).
    """)

st.caption("Construction Site AI — Generative Safety Monitoring Project — Local Execution Mode (Tier B)")
