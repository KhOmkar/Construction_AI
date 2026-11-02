import os
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Data Explorer — Construction Site AI", page_icon="📂", layout="wide")

st.title("📂 Dataset Pipeline & Worker Crop Explorer")
st.markdown("""
This module outlines the end-to-end data preparation pipeline: source ingestion, bounding box extraction, 
perceptual hash deduplication, reproducible image-level splitting, and crop extraction.
""")

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Unified Safety Class Schema")
    st.markdown("""
    All diverse raw annotation formats (YOLO txt, Pascal VOC XML, COCO JSON, CVAT XML) 
    were harmonized into a unified 5-class detection ontology:
    - **Class 0:** `helmet` (Hardhat present)
    - **Class 1:** `no_helmet` (Worker head without hardhat)
    - **Class 2:** `vest` (High-visibility reflective safety vest)
    - **Class 3:** `no_vest` (Worker torso without safety vest)
    - **Class 4:** `person` (Worker boundary)
    """)

    st.subheader("Data Preparation Ledger")
    ledger = [
        {"Stage": "1. Raw Discovery", "Count": "26,140 scenes", "Details": "8 source packages audited in data/"},
        {"Stage": "2. Staged Ingestion", "Count": "25,472 scenes", "Details": "Standardized to YOLO bounding box format"},
        {"Stage": "3. Deduplication", "Count": "6,927 duplicates removed", "Details": "64-bit dhash threshold <= 6 bits"},
        {"Stage": "4. Unique Real Scenes", "Count": "18,545 scenes", "Details": "Preserved across all 4 splits"},
        {"Stage": "5. Worker Crops (>=24px)", "Count": "33,047 crops", "Details": "Extracted at 64x64 & 128x128 into .npz"}
    ]
    st.dataframe(pd.DataFrame(ledger), use_container_width=True, hide_index=True)

with col2:
    st.subheader("Split Distributions & Benchmark Protection")
    splits_data = [
        {"Split": "Train", "Scenes": "12,788", "Crops (64x64)": "19,768", "Role": "Generative model optimization"},
        {"Split": "Val", "Scenes": "1,598", "Crops (64x64)": "2,476", "Role": "Threshold calibration & early stopping"},
        {"Split": "Test", "Scenes": "1,600", "Crops (64x64)": "2,524", "Role": "Cross-model generative evaluation"},
        {"Split": "Final Test (Untouched)", "Scenes": "2,559", "Crops (64x64)": "8,279", "Role": "Held-out real-only benchmark"}
    ]
    st.dataframe(pd.DataFrame(splits_data), use_container_width=True, hide_index=True)

    st.info("🔒 **Strict Leakage Prevention:** The `final_test` split was quarantined before any generative training began and has never been seen by any generative model or detector.")

st.divider()

st.subheader("Interactive Worker Crop Inspector")
st.markdown("Browse real 64×64 worker crops across safety compliance states from the test split:")

class_options = {
    0: "Class 0: Compliant (Hardhat & High-Vis Vest)",
    1: "Class 1: Violation — Missing Helmet",
    2: "Class 2: Violation — Missing Vest",
    3: "Class 3: Critical Violation — Both Missing"
}

selected_cid = st.selectbox("Select Worker Safety State:", options=list(class_options.keys()), format_func=lambda x: class_options[x])

test_npz_path = "processed/crops/crops_test_64.npz"
if os.path.exists(test_npz_path):
    data = np.load(test_npz_path)
    images = data["images"]
    labels = data["labels"]

    mask = (labels == selected_cid)
    matching_crops = images[mask]

    st.write(f"Found **{len(matching_crops)}** crops for {class_options[selected_cid]} in the test split.")

    if len(matching_crops) > 0:
        seed = st.slider("Sample Randomization Seed:", min_value=0, max_value=100, value=42)
        np.random.seed(seed)
        num_to_show = min(16, len(matching_crops))
        sample_indices = np.random.choice(len(matching_crops), size=num_to_show, replace=False)

        cols = st.columns(8)
        for i, idx in enumerate(sample_indices):
            col = cols[i % 8]
            crop_img = matching_crops[idx]
            col.image(crop_img, use_container_width=True, caption=f"Crop #{idx}")
else:
    st.warning("Crop file processed/crops/crops_test_64.npz not found.")
