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
st.markdown("Find your ideal career path — fully offline, open-source, and crash-proof!")
st.divider()

# ---------------------------
# LOAD MODELS
# ---------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    return embedder, generator

embedder, generator = load_models()

# ---------------------------
# LOAD LOCAL DATASET
# ---------------------------
DATA_PATH = Path("OccupationData.csv")
CACHE_PATH = Path("cached_embeddings.pt")

if not DATA_PATH.exists():
    st.error("❌ Missing OccupationData.csv in app folder.")
    st.stop()

@st.cache_data
def load_job_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.lower().strip() for c in df.columns]
    if "title" not in df.columns or "description" not in df.columns:
        st.error("Dataset must have 'title' and 'description' columns.")
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
    st.info("🔄 Computing embeddings (only first run)...")
    embeddings = embedder.encode(
        df["description"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_tensor=True
    )
    torch.save({"titles": df["title"].tolist(), "embeddings": embeddings}, CACHE_PATH)
    st.success("✅ Cached embeddings for next time.")
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

    # ---------------------------
    # GENERATE ALL 3 EXPLANATIONS
    # ---------------------------
    for i, (_, row) in enumerate(top_jobs.iterrows(), 1):
        st.markdown(f"## 🏆 #{i}. {row['title']}")
        st.caption(f"Similarity Score: {row['similarity']:.3f}")

        # Prepare structured, clear prompt
        prompt = f"""
You are an expert AI career advisor. Respond using markdown and emojis.
Provide three clear sections for the user:

### 💡 Why This Career Fits
Explain *why this career matches* the user's interests and skills in 3-5 sentences.

### 🎓 Learning Roadmap
List 4-6 specific steps or free learning methods (e.g., YouTube tutorials, MOOCs, open projects, GitHub challenges — no links). 
Be concrete: what skills or topics to learn, and in what order.

### 🚀 First Step Today
Give one small practical action the user can take right now to start.

Career Title: {row['title']}
Career Description: {row['description']}
User Interests: {interests}
User Skills: {skills}
Dream Job: {dream_job}
"""

        try:
            with st.spinner(f"Generating explanation for {row['title']}..."):
                response = generator(
                    prompt,
                    max_new_tokens=600,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95
                )[0]["generated_text"]

            st.markdown(response)
        except Exception as e:
            st.error(f"⚠️ Couldn't generate explanation for {row['title']}: {e}")

        st.divider()

    st.caption("💡 Powered by MiniLM for matching + FLAN-T5 for structured reasoning — fully offline and stable.")
