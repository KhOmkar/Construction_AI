import os
import json
import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Anomaly Detector — Construction Site AI", page_icon="🚨", layout="wide")

st.title("🚨 Autoencoder Safety Anomaly Detector")
st.markdown("""
This module uses the **Convolutional Autoencoder** trained exclusively on compliant workers ($y = 0$). 
When presented with an unseen worker crop, it reconstructs the image from its 256-dimensional bottleneck. 
Unfamiliar regions (such as a bare head or dark unvested shirt) produce high localized reconstruction error heatmaps.
""")

st.divider()

# Load anomaly metrics and threshold
eval_json_path = "runs/ae_baseline/anomaly_eval.json"
tau_star = 0.0001929
if os.path.exists(eval_json_path):
    with open(eval_json_path, "r") as f:
        metrics_dict = json.load(f)
    tau_star = float(metrics_dict.get("optimal_threshold", 0.0001929))
    val_auroc = float(metrics_dict.get("val_auroc", 0.0))
    test_auroc = float(metrics_dict.get("test_auroc", 0.0))
    test_acc = float(metrics_dict.get("test_accuracy", 0.0))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Optimal Threshold (τ*)", f"{tau_star:.6f}", "Calibrated on Val")
c2.metric("Validation AUROC", f"{val_auroc:.4f}", "Compliant vs Violation")
c3.metric("Test AUROC", f"{test_auroc:.4f}", "Held-out test split")
c4.metric("Test Classification Acc", f"{test_acc*100:.1f}%", "Zero-shot anomaly rule")

st.divider()

col_input, col_result = st.columns([1, 2])

with col_input:
    st.subheader("Test Crop Selection")
    source_mode = st.radio("Input Source:", ["Sample from Test Split", "Upload Custom Crop"])

    test_npz = "processed/crops/crops_test_64.npz"
    input_crop_tensor = None
    true_label_name = "Unknown"

    if source_mode == "Sample from Test Split":
        if os.path.exists(test_npz):
            data = np.load(test_npz)
            imgs = data["images"]
            lbls = data["labels"]

            class_labels = {
                0: "Compliant (Normal)",
                1: "Violation: Missing Helmet",
                2: "Violation: Missing Vest",
                3: "Violation: Both Missing"
            }
            c_select = st.selectbox("Select Ground-Truth Class:", options=list(class_labels.keys()), format_func=lambda x: class_labels[x])
            true_label_name = class_labels[c_select]

            matching = np.where(lbls == c_select)[0]
            crop_idx = st.slider("Select Sample Index:", 0, min(100, len(matching) - 1), 0)
            chosen_real_idx = matching[crop_idx]
            crop_np = imgs[chosen_real_idx]

            st.image(crop_np, caption=f"Selected Crop #{chosen_real_idx} ({true_label_name})", width=160)
            input_crop_tensor = torch.from_numpy(crop_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    else:
        uploaded_file = st.file_uploader("Upload 64x64 Worker Crop:", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB").resize((64, 64))
            st.image(pil_img, caption="Uploaded Crop (Resized to 64x64)", width=160)
            crop_np = np.array(pil_img)
            input_crop_tensor = torch.from_numpy(crop_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0

with col_result:
    st.subheader("Inspection & Anomaly Heatmap")

    if input_crop_tensor is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from src.models.ae import AnomalyAutoencoder

        ae_ckpt = "runs/ae_baseline/best.pt"
        if os.path.exists(ae_ckpt):
            model = AnomalyAutoencoder(in_channels=3, bottleneck_dim=256).to(device)
            ckpt = torch.load(ae_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            with torch.no_grad():
                inp = input_crop_tensor.to(device)
                recon, _ = model(inp)
                recon = recon.clamp(0, 1)

                diff = (inp - recon) ** 2
                mse_err = float(diff.mean().item())
                l1_diff = torch.abs(inp - recon).mean(dim=1).squeeze(0).cpu().numpy()

            # Create heatmap using inferno colormap
            cmap = plt.get_cmap("inferno")
            l1_norm = (l1_diff - l1_diff.min()) / (l1_diff.max() - l1_diff.min() + 1e-8)
            heatmap_rgb = (cmap(l1_norm)[:, :, :3] * 255).astype(np.uint8)

            col_a, col_b, col_c = st.columns(3)
            orig_pil = Image.fromarray((input_crop_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            recon_pil = Image.fromarray((recon.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            heat_pil = Image.fromarray(heatmap_rgb)

            col_a.image(orig_pil, caption="1. Original Worker Crop", use_container_width=True)
            col_b.image(recon_pil, caption="2. Autoencoder Reconstruction", use_container_width=True)
            col_c.image(heat_pil, caption="3. Error Heatmap", use_container_width=True)

            st.divider()

            # Verdict
            is_anomaly = (mse_err > tau_star)
            verdict_col1, verdict_col2 = st.columns([1, 1])

            with verdict_col1:
                st.metric("Crop Reconstruction MSE", f"{mse_err:.6f}", f"Threshold τ*: {tau_star:.6f}")

            with verdict_col2:
                if is_anomaly:
                    st.error("🚨 **VERDICT: SAFETY VIOLATION DETECTED**\n\nReconstruction error exceeds normal baseline threshold.")
                else:
                    st.success("🟢 **VERDICT: PPE COMPLIANT**\n\nWorker conforms to the learned safe equipment manifold.")

            st.caption(f"Ground-Truth Label: **{true_label_name}** | Decision Rule: `MSE > {tau_star:.6f}`")
        else:
            st.error("Autoencoder checkpoint runs/ae_baseline/best.pt not found.")
    else:
        st.info("Select or upload a worker crop to run real-time anomaly detection.")
