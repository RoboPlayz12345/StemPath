import streamlit as st
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="🧠 STEMPath – AI Career & Learning Guide", page_icon="🧠", layout="centered")
st.title("🧠 STEMPath – AI Career & Learning Guide")
st.markdown("Discover your ideal STEM career path using open-source AI — fully offline!")
st.divider()

# ---------------------------
# LOAD MODELS
# ---------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    generator = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
    return embedder, generator

embedder, generator = load_models()

# ---------------------------
# LOAD DATASET
# ---------------------------
DATA_PATH = Path("OccupationData.csv")
CACHE_PATH = Path("cached_embeddings.pt")

if not DATA_PATH.exists():
    st.error("❌ Missing OccupationData.csv file.")
    st.stop()

@st.cache_data
def load_jobs():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.lower().strip() for c in df.columns]
    if "title" not in df.columns or "description" not in df.columns:
        st.error("Dataset must contain 'title' and 'description' columns.")
        st.stop()
    return df.dropna(subset=["title", "description"]).reset_index(drop=True)

jobs_df = load_jobs()

# ---------------------------
# EMBEDDINGS (CACHED)
# ---------------------------
def get_embeddings(df):
    if CACHE_PATH.exists():
        cache = torch.load(CACHE_PATH)
        if len(cache["titles"]) == len(df):
            return cache["embeddings"]
    with st.spinner("🔄 Encoding job descriptions... (first time only)"):
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
    dream = st.text_input("Describe your dream job (optional):")
    submitted = st.form_submit_button("Find My Career Matches")

# ---------------------------
# MAIN LOGIC
# ---------------------------
if submitted:
    if not interests or not skills:
        st.warning("Please fill out both interests and skills.")
        st.stop()

    with st.spinner("🔍 Finding your top 3 matches..."):
        user_text = f"My interests: {interests}. My skills: {skills}. Dream job: {dream}"
        with torch.no_grad():
            user_emb = embedder.encode(user_text, convert_to_tensor=True)
            sims = util.cos_sim(user_emb, embeddings)[0]
        jobs_df["similarity"] = sims.cpu().numpy()
        top = jobs_df.sort_values("similarity", ascending=False).head(3).reset_index(drop=True)

    st.success("✅ Your Top 3 Career Recommendations")

    explanations = []
    for _, row in top.iterrows():
        prompt = f"""
You are a helpful career advisor. Explain briefly:
1. Why this job fits the user's interests and skills.
2. What topics to learn (4 short bullet points max).
3. One first step to start.

User interests: {interests}
User skills: {skills}
Dream job: {dream}
Career: {row['title']} - {row['description'][:200]}
"""
        try:
            with torch.no_grad():
                out = generator(prompt, max_new_tokens=150, do_sample=True, top_p=0.9, temperature=0.7)[0]["generated_text"]
            if len(out.strip()) < 15:
                raise ValueError("Empty response.")
        except Exception:
            out = (f"This career fits your skills in {skills} and interests in {interests}. "
                   "Start by watching beginner tutorials, joining online courses, or exploring related projects.")
        explanations.append(out)

    # ---------------------------
    # DISPLAY ALL 3 AT ONCE
    # ---------------------------
    for i, (idx, row) in enumerate(top.iterrows()):
        st.markdown(f"### 🏆 {i+1}. {row['title']}")
        st.caption(f"**Similarity Score:** {row['similarity']:.3f}")
        st.markdown(f"**Description:** {row['description'][:250]}...")
        st.markdown(explanations[i])
        st.divider()

    st.caption("💡 Fast mode: 3 parallel AI summaries using flan-t5-small")

