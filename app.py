import streamlit as st
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="🧠 STEMPath – Fast AI Career Guide", page_icon="🧠", layout="centered")
st.title("🧠 STEMPath – Fast AI Career & Learning Guide")
st.markdown("Discover your ideal STEM career — lightning fast and offline-ready.")
st.divider()

# ---------------------------
# LOAD MODELS (lightweight)
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
# EMBEDDINGS (cached)
# ---------------------------
def get_embeddings(df):
    if CACHE_PATH.exists():
        cache = torch.load(CACHE_PATH)
        if len(cache["titles"]) == len(df):
            return cache["embeddings"]
    with st.spinner("🔄 Encoding job descriptions... (1st time only)"):
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
    submitted = st.form_submit_button("Find My Matches")

# ---------------------------
# RESULTS
# ---------------------------
if submitted:
    with st.spinner("🧩 Matching careers..."):
        user_text = f"My interests: {interests}. My skills: {skills}. Dream job: {dream}"
        with torch.no_grad():
            user_emb = embedder.encode(user_text, convert_to_tensor=True)
            sims = util.cos_sim(user_emb, embeddings)[0]
        jobs_df["similarity"] = sims.cpu().numpy()
        top = jobs_df.sort_values("similarity", ascending=False).head(3)

    st.success("✅ Top Career Recommendations")

    for i, (_, row) in enumerate(top.iterrows(), 1):
        st.markdown(f"## 🏆 #{i}. {row['title']}")
        st.caption(f"Similarity: {row['similarity']:.3f}")

        # compact but still structured prompt
        prompt = f"""
You are a concise AI career mentor. In under 100 words, explain 3 things:

1️⃣ Why this career fits the user based on their interests and skills.  
2️⃣ What key topics to learn (4 short bullet points).  
3️⃣ One simple first step they can take today.

User: interests={interests}, skills={skills}, dream={dream}.  
Career: {row['title']} – {row['description']}.
"""
        try:
            with torch.no_grad():
                text = generator(prompt, max_new_tokens=200, temperature=0.7, top_p=0.9)[0]["generated_text"]
            st.markdown(text)
        except Exception as e:
            st.error(f"⚠️ Generation failed for {row['title']}: {e}")
        st.divider()

    st.caption("💡 Fast mode: using FLAN-T5-Small for instant explanations under 10s.")

