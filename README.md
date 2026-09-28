---
title: ChemoPort CT-MAR Studio
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit

---

# ChemoPort CT-MAR Studio 🩺⚡

> **3D CAD Prior-Guided Deep Learning Metal Artifact Reduction & Dosimetric Accuracy in Breast Cancer Radiation Therapy**

An open-science, interactive medical physics web platform designed for radiation oncology researchers, medical physicists, and peer-reviewers.

---

## 🌟 Key Features

1. **QuillBot-Inspired UI & FreePDFConvert-Inspired UX:**
   - Deep teal and mint dual-card layout with high-contrast medical imaging cards.
   - One-click sample loading, real-time side-by-side comparison, and direct export.
2. **CAD Prior-Guided Multi-Material Reconstruction:**
   - Multi-compartment density regularizer for Chemo Ports (Titanium casing + Silicone septum + Saline fluid chamber).
   - Inpainting of severe photon starvation dark streaks and bright flares near critical structures (skin, supraclavicular lymph nodes, ipsilateral lung).
3. **Clinical & Dosimetric Verification Tools:**
   - Real-time horizontal and radial HU line profile extraction.
   - DICOM-RT compatible re-export with HIPAA anonymization.

---

## 🚀 Running Locally

```bash
# Clone the repository
git clone https://github.com/your-username/ChemoPort-CT-MAR-Studio.git
cd ChemoPort-CT-MAR-Studio

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit application
streamlit run app.py
```

---

## 🔬 Scientific & Clinical Research Context

This studio implements the computational and clinical methodology outlined in:
* **Research Proposal:** *Prior-guided Deep Learning for Multi-material Chemo Port Metal Artifact Reduction and Dosimetric Accuracy in Breast Cancer Radiation Therapy*
* **Clinical Target:** Supraclavicular lymph node (SCN) and chest wall breast radiation therapy (VMAT / Tangential IMRT).
