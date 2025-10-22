# 🧠 STEMPath – AI Career & Learning Guide

**STEMPath** helps students discover their best-fit STEM career and a learning roadmap using a lightweight AI model.

## 🚀 Features
- Short quiz to capture user interests and skills
- AI model (FLAN-T5) suggests 3 STEM careers + learning paths
- Runs fully offline — no API keys or external calls
- Deployable on **Streamlit Cloud** (free!)

## 🧩 Tech Stack
- Python
- Streamlit (UI)
- Hugging Face Transformers (`google/flan-t5-small`)

## 🛠️ Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/stempath.git
cd stempath
pip install -r requirements.txt
streamlit run app.py
