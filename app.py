import streamlit as st
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="STEMPath – Career Discovery", page_icon="🧭", layout="centered")
st.title("🔹 STEMPath – Career Discovery Guide")
st.markdown("Discover the best STEM careers that fit your interests, skills, and goals.")
st.divider()

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

# ---------------------------
# LOAD DATASET
# ---------------------------
DATA_PATH = Path("OccupationData.csv")
CACHE_PATH = Path("cached_embeddings.pt")

if not DATA_PATH.exists():
    st.error("Missing `OccupationData.csv` in the same directory.")
    st.stop()

@st.cache_data
def load_jobs():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.lower().strip() for c in df.columns]
    if "title" not in df.columns or "description" not in df.columns:
        st.error("Dataset must include 'title' and 'description' columns.")
        st.stop()
    return df.dropna(subset=["title", "description"]).reset_index(drop=True)

jobs_df = load_jobs()

# ---------------------------
# LOAD / CACHE EMBEDDINGS
# ---------------------------
def get_embeddings(df):
    if CACHE_PATH.exists():
        cache = torch.load(CACHE_PATH)
        if len(cache["titles"]) == len(df):
            return cache["embeddings"]
    with st.spinner("Preparing dataset (only the first time)..."):
        with torch.no_grad():
            emb = embedder.encode(df["description"].tolist(), batch_size=32, convert_to_tensor=True)
        torch.save({"titles": df["title"].tolist(), "embeddings": emb}, CACHE_PATH)
    return emb

embeddings = get_embeddings(jobs_df)

# ---------------------------
# USER INPUT
# ---------------------------
with st.form("career_form"):
    st.subheader("🎯 Quick Career Quiz")
    interests = st.text_area("What topics or activities excite you most?")
    skills = st.text_area("What are your strongest skills?")
    dream = st.text_input("Describe your dream job or ideal life (optional):")
    submitted = st.form_submit_button("Find My Top 3 Careers")

# ---------------------------
# RESULTS
# ---------------------------
if submitted:
    if not interests or not skills:
        st.warning("Please fill out both interests and skills before continuing.")
        st.stop()

    with st.spinner("Analyzing your answers..."):
        user_text = f"My interests: {interests}. My skills: {skills}. Dream job: {dream}"
        with torch.no_grad():
            user_emb = embedder.encode(user_text, convert_to_tensor=True)
            sims = util.cos_sim(user_emb, embeddings)[0]
        jobs_df["similarity"] = sims.cpu().numpy()
        top = jobs_df.sort_values("similarity", ascending=False).head(3).reset_index(drop=True)

    st.success("Your Top 3 Career Matches")
    for i, row in top.iterrows():
        st.markdown(f"### {i+1}. {row['title']}")
        st.caption(f"Match Score: {row['similarity']:.3f}")
        st.markdown(f"{row['description'][:400]}...")
        st.divider()

    st.caption("STEMPath uses semantic search to match your profile to real-world careers.")
