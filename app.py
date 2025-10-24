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
st.markdown("Find your ideal career path using open-source AI — fully offline and instant results!")
st.divider()

# ---------------------------
# LOAD MODELS
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    generator = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
    return embedder, generator
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder, generator = load_models()
embedder = load_embedder()

# ---------------------------
# LOAD DATASET
@@ -31,7 +28,7 @@ def load_models():
CACHE_PATH = Path("cached_embeddings.pt")

if not DATA_PATH.exists():
    st.error("❌ Missing OccupationData.csv file.")
    st.error("❌ Missing OccupationData.csv file in the same directory.")
    st.stop()

@st.cache_data
@@ -46,7 +43,7 @@ def load_jobs():
jobs_df = load_jobs()

# ---------------------------
# EMBEDDINGS (CACHED)
# LOAD / CACHE EMBEDDINGS
# ---------------------------
def get_embeddings(df):
    if CACHE_PATH.exists():
@@ -62,24 +59,24 @@ def get_embeddings(df):
embeddings = get_embeddings(jobs_df)

# ---------------------------
# USER INPUT
# USER INPUT FORM
# ---------------------------
with st.form("career_form"):
    st.subheader("🎯 Quick Career Quiz")
    interests = st.text_area("What topics or activities excite you most?")
    skills = st.text_area("What are your strongest skills?")
    dream = st.text_input("Describe your dream job (optional):")
    submitted = st.form_submit_button("Find My Career Matches")
    dream = st.text_input("Describe your dream job or ideal life (optional):")
    submitted = st.form_submit_button("Find My Top 3 Careers")

# ---------------------------
# MAIN LOGIC
# RESULTS
# ---------------------------
if submitted:
    if not interests or not skills:
        st.warning("Please fill out both interests and skills.")
        st.stop()

    with st.spinner("🔍 Finding your top 3 matches..."):
    with st.spinner("🔍 Matching careers..."):
        user_text = f"My interests: {interests}. My skills: {skills}. Dream job: {dream}"
        with torch.no_grad():
            user_emb = embedder.encode(user_text, convert_to_tensor=True)
@@ -88,39 +85,12 @@ def get_embeddings(df):
        top = jobs_df.sort_values("similarity", ascending=False).head(3).reset_index(drop=True)

    st.success("✅ Your Top 3 Career Recommendations")
    st.markdown("Here are the careers that best match your profile:")

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
    for i, row in top.iterrows():
        st.markdown(f"### 🏆 {i+1}. {row['title']}")
        st.caption(f"**Similarity Score:** {row['similarity']:.3f}")
        st.markdown(f"**Description:** {row['description'][:250]}...")
        st.markdown(explanations[i])
        st.markdown(f"**Description:** {row['description'][:400]}...")
        st.divider()

    st.caption("💡 Fast mode: 3 parallel AI summaries using flan-t5-small")

    st.caption("💡 Fast mode enabled — powered by MiniLM embeddings for instant matching.")
