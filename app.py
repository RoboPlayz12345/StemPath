import streamlit as st
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="🧠 STEMPath – AI Career & Learning Guide",
    page_icon="🧠",
    layout="centered"
)
st.title("🧠 STEMPath – AI Career & Learning Guide")
st.markdown("Discover your ideal career path using open-source AI — no API keys, fully offline!")
st.divider()

# ---------------------------
# Load Models
# ---------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    return embedder, generator

embedder, generator = load_models()

# ---------------------------
# Upload Dataset
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload your O*NET or job dataset (CSV/Excel) with 'title' and 'description' columns",
    type=["csv", "xlsx"]
)

CACHE_PATH = Path("cached_embeddings.pt")

def load_job_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df.columns = [c.lower().strip() for c in df.columns]
    if "title" not in df.columns or "description" not in df.columns:
        st.error("Dataset must have 'title' and 'description' columns.")
        st.stop()
    return df.dropna(subset=["title", "description"]).reset_index(drop=True)

def compute_embeddings(df):
    # Use cached embeddings if available
    if CACHE_PATH.exists():
        cache = torch.load(CACHE_PATH)
        if len(cache["titles"]) == len(df):
            st.info("✅ Loaded cached embeddings.")
            return cache["embeddings"]
    st.info("🔄 Computing embeddings (may take a few minutes)...")
    embeddings = embedder.encode(df["description"].tolist(), batch_size=32, show_progress_bar=True, convert_to_tensor=True)
    torch.save({"titles": df["title"].tolist(), "embeddings": embeddings}, CACHE_PATH)
    st.success("✅ Embeddings cached for future runs.")
    return embeddings

if uploaded_file:
    jobs_df = load_job_data(uploaded_file)
    embeddings = compute_embeddings(jobs_df)
else:
    st.warning("Please upload a dataset to continue.")
    st.stop()

# ---------------------------
# Quiz Form
# ---------------------------
with st.form("career_form"):
    st.subheader("🎯 Quick Career Quiz")
    interests = st.text_area("What topics or activities excite you most?")
    skills = st.text_area("What are your strongest skills?")
    dream_job = st.text_input("Describe your dream job or ideal life (optional):")
    submitted = st.form_submit_button("Find My Best Career Matches")

# ---------------------------
# Matching
# ---------------------------
if submitted:
    with st.spinner("🧠 Finding best career matches..."):
        user_text = f"My interests: {interests}. My skills: {skills}. Dream job: {dream_job}."
        user_emb = embedder.encode(user_text, convert_to_tensor=True)

        sims = util.cos_sim(user_emb, embeddings)[0]
        jobs_df["similarity"] = sims.cpu().numpy()
        top_jobs = jobs_df.sort_values("similarity", ascending=False).head(3)

    st.success("✅ Your Top Career Recommendations")

    # ---------------------------
    # Generate Friendly Explanations
    # ---------------------------
    for _, row in top_jobs.iterrows():
        st.markdown(f"### 🏆 {row['title']}")
        st.caption(f"Similarity Score: {row['similarity']:.3f}")

        prompt = f"""
You are a friendly AI career advisor. Explain in detail why this career is a great fit for the user,
and suggest how to start learning it.

Career: {row['title']}
Description: {row['description']}
User Interests: {interests}
User Skills: {skills}
Dream Job: {dream_job}

Include:
1. Why this fits the user
2. Learning roadmap (free resources, no links)
3. First step to start today
"""

        response = generator(prompt, max_new_tokens=400, do_sample=True)[0]["generated_text"]
        st.markdown(response)
        st.divider()

    st.caption("💡 Powered by MiniLM for matching + FLAN-T5 for explanations — fully offline, crash-proof.")
