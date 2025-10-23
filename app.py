import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from pathlib import Path
import os

# --------------------------------------
# STREAMLIT PAGE SETTINGS
# --------------------------------------
st.set_page_config(
    page_title="STEMPath – AI Career & Learning Guide",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 STEMPath – AI Career & Learning Guide")
st.markdown("Discover your ideal career path using open-source AI — no API keys, no limits!")
st.divider()


# --------------------------------------
# LOAD MODELS (cached)
# --------------------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    return embedder, generator

embedder, generator = load_models()


# --------------------------------------
# LOAD OR UPLOAD O*NET-LIKE DATASET
# --------------------------------------
st.subheader("📁 Upload Your O*NET / Job Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file containing at least 'title' and 'description' columns",
    type=["csv", "xlsx"]
)

# Path for cached embeddings
CACHE_PATH = Path("cached_job_embeddings.pt")

# --------------------------------------
# LOAD DATASET FUNCTION
# --------------------------------------
def load_job_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]
    if "title" not in df.columns or "description" not in df.columns:
        st.error("Dataset must have columns named 'title' and 'description'.")
        st.stop()
    df = df.dropna(subset=["title", "description"]).reset_index(drop=True)
    return df


# --------------------------------------
# ENCODING & CACHING
# --------------------------------------
def compute_embeddings(df):
    """Compute or load embeddings safely for large datasets."""
    if CACHE_PATH.exists():
        cache = torch.load(CACHE_PATH)
        if len(cache["titles"]) == len(df):
            st.info("✅ Loaded cached embeddings.")
            return cache["embeddings"]
        else:
            st.warning("⚠️ Dataset changed — re-computing embeddings.")

    st.info("🔄 Computing embeddings for all job descriptions (first time only, please wait)...")
    texts = df["description"].tolist()
    # Batched encoding for large data
    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True, convert_to_tensor=True)
    torch.save({"titles": df["title"].tolist(), "embeddings": embeddings}, CACHE_PATH)
    st.success("✅ Embeddings saved for future use.")
    return embeddings


if uploaded_file:
    jobs_df = load_job_data(uploaded_file)
    embeddings = compute_embeddings(jobs_df)
else:
    st.warning("Please upload your dataset to continue.")
    st.stop()


# --------------------------------------
# QUIZ FORM
# --------------------------------------
with st.form("career_form"):
    st.subheader("🎯 Quick Career Quiz")
    interests = st.text_area("What topics or activities excite you most?")
    skills = st.text_area("What are your strongest skills?")
    dream_job = st.text_input("Describe your dream job or lifestyle (optional):")
    submitted = st.form_submit_button("Find My Best Career Matches")


# --------------------------------------
# MATCHING LOGIC
# --------------------------------------
if submitted:
    with st.spinner("🧠 Analyzing your profile and finding best career matches..."):
        user_text = f"My interests: {interests}. My skills: {skills}. My dream job: {dream_job}."
        user_embedding = embedder.encode(user_text, convert_to_tensor=True)

        # Compute cosine similarities
        sims = util.cos_sim(user_embedding, embeddings)[0]
        jobs_df["similarity"] = sims.cpu().numpy()

        # Top 3 matches
        top_jobs = jobs_df.sort_values("similarity", ascending=False).head(3)

    st.success("✅ Your Top Career Matches")

    # --------------------------------------
    # GENERATE AI EXPLANATIONS
    # --------------------------------------
    for _, row in top_jobs.iterrows():
        st.markdown(f"### 🏆 {row['title']}")
        st.caption(f"Similarity Score: {row['similarity']:.3f}")

        prompt = f"""
You are a friendly AI career advisor.
Explain why the following career is a great fit for someone with these traits,
and suggest how they can begin learning it.

Career: {row['title']}
Description: {row['description']}
User Interests: {interests}
User Skills: {skills}
Dream Job: {dream_job}

Include:
1. Why this fits the user
2. Learning Roadmap (free resources, no links)
3. First Step to Start Today
"""
        response = generator(prompt, max_new_tokens=400, temperature=0.7)[0]["generated_text"]
        st.markdown(response)
        st.divider()

    st.caption("💡 Powered by MiniLM for matching + FLAN-T5 for explanations — runs fully local.")
