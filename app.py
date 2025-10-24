import streamlit as st
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ---------------------------
# PAGE CONFIG
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
# LOAD MODELS
# ---------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    try:
        generator = pipeline("text2text-generation", model="google/flan-t5-large")
    except Exception:
        st.warning("⚠️ Using smaller fallback model: flan-t5-base")
        generator = pipeline("text2text-generation", model="google/flan-t5-base")
    return embedder, generator

embedder, generator = load_models()

# ---------------------------
# LOAD LOCAL DATASET
# ---------------------------
DATA_PATH = Path("OccupationData.csv")
CACHE_PATH = Path("cached_embeddings.pt")

if not DATA_PATH.exists():
    st.error("❌ Could not find OccupationData.csv in app directory.")
    st.stop()

@st.cache_data
def load_job_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.lower().strip() for c in df.columns]
    if "title" not in df.columns or "description" not in df.columns:
        st.error("Dataset must include 'title' and 'description' columns.")
        st.stop()
    return df.dropna(subset=["title", "description"]).reset_index(drop=True)

jobs_df = load_job_data()

# ---------------------------
# EMBEDDINGS (cached)
# ---------------------------
def compute_embeddings(df):
    if CACHE_PATH.exists():
        cache = torch.load(CACHE_PATH)
        if len(cache["titles"]) == len(df):
            st.info("✅ Loaded cached embeddings.")
            return cache["embeddings"]
    st.info("🔄 Computing embeddings (first run may take a few minutes)...")
    embeddings = embedder.encode(
        df["description"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_tensor=True
    )
    torch.save({"titles": df["title"].tolist(), "embeddings": embeddings}, CACHE_PATH)
    st.success("✅ Cached embeddings for faster future runs.")
    return embeddings

embeddings = compute_embeddings(jobs_df)

# ---------------------------
# QUIZ FORM
# ---------------------------
with st.form("career_form"):
    st.subheader("🎯 Quick Career Quiz")
    interests = st.text_area("What topics or activities excite you most?")
    skills = st.text_area("What are your strongest skills?")
    dream_job = st.text_input("Describe your dream job or ideal life (optional):")
    submitted = st.form_submit_button("Find My Best Career Matches")

# ---------------------------
# MATCHING & RESULTS
# ---------------------------
if submitted:
    with st.spinner("🧠 Finding your best matches..."):
        user_text = f"My interests: {interests}. My skills: {skills}. Dream job: {dream_job}."
        user_emb = embedder.encode(user_text, convert_to_tensor=True)
        sims = util.cos_sim(user_emb, embeddings)[0]
        jobs_df["similarity"] = sims.cpu().numpy()
        top_jobs = jobs_df.sort_values("similarity", ascending=False).head(3)

    st.success("✅ Your Top Career Recommendations")

    for _, row in top_jobs.iterrows():
        st.markdown(f"### 🏆 {row['title']}")
        st.caption(f"Similarity Score: {row['similarity']:.3f}")

        prompt = f"""
You are a friendly expert AI career advisor. 
You must clearly output the following 3 sections in detail with emojis and markdown formatting:

### 💡 Why This Career Fits
Explain exactly why this career matches the user's interests and skills, using specific reasoning.

### 🎓 Learning Roadmap
List 4–6 realistic steps or resources (free YouTube, MOOCs, open projects, GitHub practice ideas — no links). 
Mention what to study in order and what skills to build.

### 🚀 First Step Today
Give one easy actionable step they can take right now to begin their journey.

Career Title: {row['title']}
Career Description: {row['description']}
User Interests: {interests}
User Skills: {skills}
Dream Job: {dream_job}

Be warm, detailed, and inspiring.
"""
        response = generator(
            prompt,
            max_new_tokens=600,
            temperature=0.9,
            top_p=0.95,
            do_sample=True
        )[0]["generated_text"]

        st.markdown(response)
        st.divider()

    st.caption("💡 Powered by MiniLM for matching + FLAN-T5 for detailed reasoning — 100% offline, safe, and optimized.")
