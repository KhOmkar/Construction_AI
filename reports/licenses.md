# Dataset Licenses and Compliance Audit

This document records the intellectual property, copyright, and licensing conditions for each raw dataset discovered in `data/`, along with risk assessments for university research and downstream deployment.

---

## 1. Summary Matrix

| Dataset / File | Claimed Source & Author | License Type | Commercial Use | Derivative Works / Copyleft | Compliance Risk Level |
|---|---|---|---|---|---|
| `20250731-PPE2286y.zip` | Roboflow Universe (`ying2001`) | **CC BY 4.0** | Permitted | Permitted (Attribution required) | **Low** |
| `archive.zip` | Roboflow Universe (`echo-dotf3`, Safety Vests) | **CC BY 4.0** | Permitted | Permitted (Attribution required) | **Low** |
| `css-data.zip` | Roboflow Universe (`construction-site-safety` v28) | **CC BY 4.0** | Permitted | Permitted (Attribution required) | **Low** |
| `PPE Dataset for Workplace Safety...zip` | Roboflow Universe (`siabar`) | **CC BY 4.0** | Permitted | Permitted (Attribution required) | **Low** |
| `PPE.zip` | Roboflow Universe (`tanish-y7iqo`) | **CC BY 4.0** | Permitted | Permitted (Attribution required) | **Low** |
| `data/test`, `data/train` (`jhboyo/ppe-dataset`) | Hugging Face (`jhboyo`) | **MIT License** | Permitted | Permitted (Attribution required) | **Very Low** |
| `archive (1).zip` | CVAT Export (`TrainingData`) | **Unspecified / Open** | Clarification advised | Permitted for academic research | **Medium-Low** |
| `construction-ppe.ndjson` | Ultralytics (`Dalvi et al., 2025`) | **AGPL-3.0 / Ultralytics License** | Restricted under AGPL | **Strong Copyleft** | **HIGH** |

---

## 2. In-Depth License Analysis & Risk Notes

### A. Ultralytics Construction-PPE (`construction-ppe.ndjson`) — **HIGH RISK (AGPL-3.0)**
- **Origin**: Curated and published by Ultralytics Inc. (Dalvi, Singh, Bhingarde, and Chalke, 2025) at `docs.ultralytics.com/datasets/detect/construction-ppe`.
- **License**: Ultralytics releases its open-source weights and datasets under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
- **Risk Assessment**:
  - AGPL-3.0 is a strong copyleft license. If models trained on AGPL data are used in a networked service, pipeline, or distributed project, the complete source code and modifications must also be released under AGPL-3.0.
  - For academic and university thesis evaluation, AGPL-3.0 is generally permissible provided the thesis and code remain open-source. However, if this project is ever commercialized or integrated into proprietary workplace safety software, AGPL-3.0 prevents proprietary distribution without an Ultralytics commercial license.
  - **Operational Status**: The local file is an NDJSON with 1,416 signed CDN URLs. No local images have been fetched yet. Because we already have **23,436 real images across other CC-BY and MIT datasets**, including this dataset is optional and subject to explicit user approval.

### B. Roboflow Universe Datasets (`20250731-PPE2286y`, `archive.zip`, `css-data.zip`, `PPE Dataset...`, `PPE.zip`) — **LOW RISK (CC BY 4.0)**
- **Origin**: Exported from Roboflow Universe public projects.
- **License**: **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
- **Terms**: Permits sharing, adaptation, and commercial use, provided appropriate credit is given and license link is provided.
- **Attribution Obligation**: In the project thesis/report, each Roboflow project URL and original uploader must be cited in the references.

### C. Hugging Face Dataset (`jhboyo/ppe-dataset`) — **VERY LOW RISK (MIT)**
- **Origin**: Hugging Face dataset repository `jhboyo/ppe-dataset`.
- **License**: **MIT License**.
- **Terms**: Permissive, standard software/data license allowing arbitrary modification, distribution, and use.

### D. CVAT Helmet Annotations (`archive (1).zip`) — **MEDIUM-LOW RISK**
- **Origin**: Exported from CVAT (Computer Vision Annotation Tool) with metadata owner `TrainingData`.
- **License**: No explicit `LICENSE` or `README` bundled. Contains 46 images with 83 helmet/no_helmet annotations.
- **Recommendation**: Suitable for educational/research use. Given the tiny size (46 images), it provides modest value; user may choose to include it or exclude it to keep provenance 100% clean.

---

## 3. Worker Privacy & Ethical Considerations
- Construction worker images capture real human beings on active job sites.
- In accordance with ethical guidelines:
  1. Any public demos, web interfaces, or published figures will blur or crop identifiable facial features.
  2. The primary generative models operate on cropped worker patches (64x64 / 128x128), which inherently obfuscates facial identity while retaining safety equipment context (hard hats, vests).
