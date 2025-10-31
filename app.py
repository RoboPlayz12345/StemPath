import streamlit as st
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from gtts import gTTS
import io

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="STEMPath+ – AI Career Discovery", page_icon="🧭", layout="centered")
st.title("🧭 STEMPath+ – AI Career Discovery Guide")
st.markdown("Discover your ideal STEM career path using smart AI recommendations — offline or enhanced with AI APIs.")
st.divider()

# -----------------------------
# LOAD MODEL (OFFLINE)
# -----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

# -----------------------------
# LOAD DATASET
# -----------------------------
DATA_PATH = Path("OccupationData.csv")

if not DATA_PATH.exists():
    st.error("❌ Missing `OccupationData.csv` in the same directory.")
    st.stop()

@st.cache_data
def load_jobs():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.lower().strip() for c in df.columns]  # Normalize
    if "title" not in df.columns or "description" not in df.columns:
        st.error("Dataset must include 'title' and 'description' columns.")
        st.stop()
    return df.dropna(subset=["title", "description"]).reset_index(drop=True)

jobs_df = load_jobs()

# -----------------------------
# EMBEDDINGS (CACHED IN APP)
# -----------------------------
@st.cache_resource
def get_embeddings(df):
    with torch.no_grad():
        return embedder.encode(df["description"].tolist(), batch_size=32, convert_to_tensor=True)

embeddings = get_embeddings(jobs_df)

# -----------------------------
# SIDEBAR – OPTIONAL AI SETTINGS
# -----------------------------
st.sidebar.header("⚙️ AI Integration (Optional)")
ai_provider = st.sidebar.selectbox(
    "Choose AI provider:",
    ["Offline Mode", "OpenAI", "Anthropic", "Google Gemini", "Hugging Face"]
)
api_key = st.sidebar.text_input("Enter your API Key (optional):", type="password")
use_api = api_key.strip() != "" and ai_provider != "Offline Mode"

if use_api:
    st.sidebar.success(f"✅ Using {ai_provider} for enhanced insights.")
else:
    st.sidebar.info("Running in Offline Mode – base recommendations only.")

# -----------------------------
# TEXT-TO-SPEECH FUNCTION
# -----------------------------
def speak_text(text):
    try:
        tts = gTTS(text)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        st.audio(audio_bytes.read(), format="audio/mp3")
    except Exception as e:
        st.warning(f"Speech error: {e}")

# -----------------------------
# AI ENHANCEMENT FUNCTION
# -----------------------------
def generate_summary(provider, key, career_title, description, user_text):
    prompt = (
        f"You are a STEM career advisor. The user described: {user_text}\n"
        f"Career: {career_title}\n"
        f"Description: {description}\n"
        f"Give a short personalized explanation (3–4 sentences) about why this fits them "
        f"and one practical next step they can take."
    )

    try:
        if provider == "OpenAI":
            import openai
            openai.api_key = key
            res = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return res.choices[0].message.content.strip()

        elif provider == "Anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=key)
            msg = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            return msg.content[0].text.strip()

        elif provider == "Google Gemini":
            import google.generativeai as genai
            genai.configure(api_key=key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text.strip()

        elif provider == "Hugging Face":
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=key)
            res = client.text_generation(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                prompt=prompt,
                max_new_tokens=200
            )
            return res.strip()

    except Exception:
        return "⚠️ API error or invalid key. Please check your API credentials and try again."

# -----------------------------
# USER INPUT FORM
# -----------------------------
with st.form("career_form"):
    st.subheader("🎯 Quick Career Quiz")
    interests = st.text_area("What topics or activities excite you most?")
    skills = st.text_area("What are your strongest skills?")
    dream = st.text_input("Describe your dream job or ideal life (optional):")
    submitted = st.form_submit_button("Find My Top 3 Careers")

# -----------------------------
# RESULTS
# -----------------------------
if submitted:
    if not interests or not skills:
        st.warning("Please fill out both interests and skills before continuing.")
        st.stop()

    with st.spinner("Analyzing your responses..."):
        user_text = f"My interests: {interests}. My skills: {skills}. Dream job: {dream}"
        with torch.no_grad():
            user_emb = embedder.encode(user_text, convert_to_tensor=True)
            sims = util.cos_sim(user_emb, embeddings)[0]
        jobs_df["similarity"] = sims.cpu().numpy()
        top = jobs_df.sort_values("similarity", ascending=False).head(3).reset_index(drop=True)

    st.success("🎓 Your Top 3 Career Matches")
    for i, row in top.iterrows():
        st.markdown(f"### {i+1}. {row['title']}")
        st.caption(f"Match Score: {row['similarity']:.3f}")
        st.markdown(f"{row['description'][:400]}...")

        # Optional AI enhancement
        if use_api:
            with st.spinner(f"Enhancing with {ai_provider}..."):
                summary = generate_summary(ai_provider, api_key, row["title"], row["description"], user_text)
            st.markdown(f"🧠 **AI Insight:** {summary}")
        else:
            summary = f"This career aligns with your interests and skills. Explore online courses and internships in {row['title']} to begin your journey."
            st.markdown(f"💡 **Suggestion:** {summary}")

        # Speaker button
        col1, col2 = st.columns([1,5])
        if col1.button("🔊 Speak", key=f"speaker_{i}"):
            speak_text(f"{row['title']}. {summary}")

        st.divider()

    st.caption("STEMPath+ uses AI embeddings for offline career matching and optional API models for deeper insights.")
