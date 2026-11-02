import streamlit as st

st.set_page_config(page_title="About & Ethics — Construction Site AI", page_icon="ℹ️", layout="wide")

st.title("ℹ️ Project Architecture, Ethics & Licensing")
st.markdown("""
**Construction Site AI: Workplace Safety Monitoring via Generative Machine Learning**  
A comprehensive empirical investigation of deep generative paradigms on industrial safety monitoring.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Ethics & Worker Privacy")
    st.markdown("""
    Workplace safety monitoring involves processing visual feeds of human workers on active job sites. 
    To protect individual dignity and privacy rights, this project enforces strict safeguards:
    - **Face De-Identification:** Visual features are cropped strictly at the worker torso and hardhat level, avoiding high-resolution facial biometrics.
    - **No Surveillance for Productivity:** The model detects PPE compliance (hard-hat and high-visibility vest presence) strictly for occupational health and hazard mitigation, not worker tracking.
    - **Public Demo Safeguards:** All sample images displayed in public galleries feature blurred facial features or synthetic crops where identity is non-existent.
    """)

    st.subheader("Local Execution & Sustainability")
    st.markdown("""
    All training and evaluation pipelines operate strictly on a local consumer laptop:
    - **GPU:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (Tier B, 95W TGP)
    - **Memory Optimization:** Mixed-precision FP16, gradient accumulation, and CPU offloading during transformer training kept peak VRAM $< 350\\text{ MB}$ for ConvNets and $< 3\\text{ GB}$ for Diffusion.
    - **Energy Efficiency:** Full 100-epoch training cycles require $< 15$ minutes per model, minimizing carbon footprint.
    """)

with col2:
    st.subheader("Data Licensing & IP Governance")
    st.markdown("""
    The dataset was curated from multiple public PPE sources with careful IP compliance:
    - **JH-Boyo PPE Dataset:** Creative Commons Attribution (CC-BY 4.0)
    - **Kaggle Safety Vests:** Public Domain / Open Source (MIT License)
    - **Roboflow Construction PPE:** Academic Research License
    - **Risk Mitigation:** `data/construction-ppe.ndjson` was audited and excluded to eliminate copyleft AGPL-3.0 obligations, ensuring complete academic and commercial safety.
    """)

    st.subheader("System Architecture")
    st.code("""
+-------------------------------------------------------------+
|               Unified Pipeline Architecture                 |
+-------------------------------------------------------------+
| Raw Scenes (YOLO/VOC/COCO) -> Deduplication (dhash 64-bit)  |
|         -> Seeded Splits (80/10/10 + Held-out Test)         |
|         -> 64x64 uint8 Crops (.npz compressed arrays)       |
+-------------------------------------------------------------+
                              |
       +----------------------+----------------------+
       |                      |                      |
[Unsupervised]         [Supervised Gen]        [Adversarial]
1. Autoencoder         2. Conditional VAE      3. Conditional GAN
(Compliant only)       (Smooth Manifold)       (DiffAugment)
       |                      |                      |
       +----------------------+----------------------+
                              |
              +---------------+---------------+
              |                               |
       [Diffusion]                     [Transformer]
       4. DDPM / DDIM                  5. VQ-VAE + GPT
       (Classifier-Free Guidance)      (Discrete Tokens)
                              |
+-------------------------------------------------------------+
|         Cross-Model Evaluation (FID, KID, CCA, LPIPS)       |
|                 Streamlit Interactive Dashboard             |
+-------------------------------------------------------------+
    """, language="text")

st.caption("Construction Site AI — Academic Research Project — Developed with Antigravity")
