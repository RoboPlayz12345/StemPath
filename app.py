import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from gtts import gTTS
import io
from pathlib import Path

# Optional AI imports
import openai
import anthropic
import google.generativeai as genai
from huggingface_hub import InferenceClient

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="STEMPath – AI Career Discovery", page_icon="🧭", layout="centered")
st.title("🧭 STEMPath – AI Career Discovery")
st.markdown("Find your best-fit STEM career path using AI-powered semantic matching and insights.")
st.divider()

# ---------------------------
# LOAD DATASET
# ---------------------------
DATA_PATH = Path("OccupationData.csv")
CACHE_PATH = Path("cached_embeddings.pt")
try:
    df_jobs = pd.read_csv(DATA_PATH).dropna(subset=["Title", "Description"]).reset_index(drop=True)
except FileNotFoundError:
    st.error("❌ Missing `OccupationData.csv` in the same directory.")
    st.stop()

# ---------------------------
# LOAD EMBEDDER
# ---------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

# ---------------------------
# TEXT-TO-SPEECH
# ---------------------------
def speak_text(text):
    try:
        tts = gTTS(text)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        st.audio(audio_bytes.getvalue(), format="audio/mp3")
    except Exception as e:
        st.warning(f"Speech synthesis failed: {e}")

# ---------------------------
# OPTIONAL API KEY
# ---------------------------
st.subheader("🔑 Optional Personalization")
api_brand = st.selectbox(
    "Choose your AI provider (optional):",
    ["None", "OpenAI", "Gemini", "Anthropic", "Hugging Face"]
)
api_key = st.text_input("Enter your API key (optional):", type="password")

# Try to validate API key
valid_api = False
if api_brand != "None" and api_key:
    try:
        if api_brand == "OpenAI":
            openai.api_key = api_key
            openai.models.list()
        elif api_brand == "Gemini":
            genai.configure(api_key=api_key)
            genai.list_models()
        elif api_brand == "Anthropic":
            client = anthropic.Anthropic(api_key=api_key)
            client.models.list()
        elif api_brand == "Hugging Face":
            client = InferenceClient(token=api_key)
            _ = client.text_generation("test")
        st.success(f"✅ Connected to {api_brand} API successfully.")
        valid_api = True
    except Exception:
        st.error(f"❌ Your {api_brand} API key is incorrect. Continuing in base mode.")
else:
    st.info("Running in base mode (no external API).")

st.divider()

# ---------------------------
# USER INPUT FORM
# ---------------------------
with st.form("career_form"):
    st.subheader("🎯 Quick Career Quiz")
    interests = st.text_area("What topics or activities excite you most?")
    skills = st.text_area("What are your strongest skills?")
    dream = st.text_input("Describe your dream job or ideal life (optional):")
    submitted = st.form_submit_button("Find My Top 3 Careers")

# ---------------------------
# PROCESS RESULTS
# ---------------------------
if submitted:
    if not interests or not skills:
        st.warning("Please fill out both interests and skills before continuing.")
        st.stop()

    user_text = f"My interests: {interests}. My skills: {skills}. Dream job: {dream}"

    with st.spinner("Analyzing your responses..."):
        with torch.no_grad():
            user_emb = embedder.encode(user_text, convert_to_tensor=True)
            embeddings = embedder.encode(df_jobs["description"].tolist(), convert_to_tensor=True)
            sims = util.cos_sim(user_emb, embeddings)[0]

        df_jobs["similarity"] = sims.cpu().numpy()
        top = df_jobs.sort_values("similarity", ascending=False).head(3).reset_index(drop=True)

    st.success("Your Top 3 Career Matches 🎓")
    for i, row in top.iterrows():
        st.markdown(f"### {i+1}. {row['title']}")
        st.caption(f"Match Score: {row['similarity']:.3f}")
        st.markdown(f"{row['description'][:400]}...")
        if st.button(f"🔊 Speak career #{i+1}", key=f"tts_{i}"):
            speak_text(row["description"])
        st.divider()

    # Optional: Personalized AI summary if valid API
    if valid_api:
        st.subheader("✨ Personalized AI Career Insights")
        prompt = f"Based on these interests and skills: {user_text}, summarize why these 3 STEM careers might fit well."
        try:
            summary = ""
            if api_brand == "OpenAI":
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                summary = response.choices[0].message.content
            elif api_brand == "Gemini":
                model = genai.GenerativeModel("gemini-1.5-flash")
                summary = model.generate_content(prompt).text
            elif api_brand == "Anthropic":
                summary = client.messages.create(
                    model="claude-3",
                    messages=[{"role": "user", "content": prompt}]
                ).content[0].text
            elif api_brand == "Hugging Face":
                summary = client.text_generation(prompt, model="mistralai/Mistral-7B-Instruct-v0.2", max_new_tokens=200)
            st.markdown(f"🧩 **AI Insight:** {summary}")
        except Exception as e:
            st.warning(f"⚠️ AI insight unavailable: {e}")

st.caption("STEMPath uses semantic search to match your profile with real-world STEM careers.")




