# 🧠 NeuroScan AI: Brain Tumor Classification with Swin Transformer

**Developed by Vikash Kumar Adhikari**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://neuroscan-ai.streamlit.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

NeuroScan AI is a high-precision medical imaging project designed to classify brain tumors from axial T1-weighted MRI scans. Utilizing state-of-the-art vision architectures and modern AI diagnostic tools, this application provides rapid visual classification and AI-driven medical insights.

---

## 🚀 Project Overview

This project implements a fine-tuned **Swin Transformer (Shifted Windows)** model, which has consistently outperformed traditional Convolutional Neural Networks (CNNs) in medical vision tasks by capturing both local and global dependencies in image data.

### 🔬 Key Features
- **Accurate Classification**: Identifies Glioma, Meningioma, Pituitary tumors, or Healthy brains with **96.47% accuracy**.
- **Brutalist Web UI**: A clean, bold, and modern web interface built with Streamlit, optimized for both light and dark modes.
- **AI Diagnosis Engine**: Integrated with the **Groq Llama-3 API** to provide detailed medical descriptions, typical symptoms, prognosis, and common medical assistance for each detection.
- **Standalone CLI**: A robust Python script for researchers and developers to run batch inference.

---

## 🛠️ Technical Stack

- **Model Architecture**: [Swin-Tiny Patch4-Window7-224](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)
- **Deep Learning Framework**: PyTorch 2.1.0+
- **Inference & UI**: Streamlit 1.28.0
- **Medical Intelligence**: Groq API (Llama-3.3-70b-versatile)
- **Image Processing**: Hugging Face `transformers` ViTImageProcessor

---

## 📂 Dataset & Training
The model was fine-tuned on a contrast-enhanced MRI dataset containing 4 distinct classes:
1.  **Glioma Tumor**: Highly aggressive primary brain tumor originating in glial cells.
2.  **Meningioma Tumor**: Usually benign tumors that arise from the meninges (membranes covering brain/spinal cord).
3.  **Pituitary Tumor**: Abnormal growths on the pituitary gland.
4.  **No Tumor**: Healthy MRI scans with no detectable neural masses.

**Training Performance:**
- **Recall/Precision**: 96%+ across all classes.
- **Final Validation Loss**: 0.1012

---

## 📥 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Installation
Clone the project and install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### A. Running the Web Application (Streamlit)
To launch the interactive NeuroScan interface:
```bash
streamlit run app.py
```
This will open a local web server (usually at `http://localhost:8501`). Simply upload an MRI scan to begin analysis.

### B. Running Standalone Inference (CLI)
For quick classification of individual files:
```bash
python run_test.py "path/to/your/image.jpg"
```

---

## 🩺 AI Medical Insights
Once a tumor is detected, the application leverages the **Groq Cloud API** to generate a neuro-medical summary including:
- **Diagnosis Overview**: What the specific tumor type entails.
- **Symptoms**: Common neurological signs associated with the diagnosis.
- **Medical Assistance**: Typical medications and management protocols.
- **Disclaimer**: All AI-generated insights must be verified by a board-certified neurologist.

---

## 📜 License & Acknowledgments
- **Developer**: Devanshu Mahato
- **Architecture Source**: Microsoft Swin Transformer
- **Medical Disclosure**: This project is for educational and research purposes only. It is not intended for clinical use without proper medical certification.

---

**Made by Vikash Kumar Adhikari**
