import streamlit as st
import pandas as pd
from transformers import pipeline

st.set_page_config(
    page_title="🧠 STEMPath – AI Career & Learning Guide",
    page_icon="🧭",
    layout="centered"
)

st.title("🧠 STEMPath – AI Career & Learning Guide")
st.markdown("Discover your ideal career path powered by open-source AI — no APIs, 100% open-source!")

st.divider()

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="MBZUAI/LaMini-Flan-T5-248M", device_map="auto")

if "model" not in st.session_state:
    st.session_state["model"] = load_model()
model = st.session_state["model"]

# -----------------------------
# LOAD CAREER DATASET
# -----------------------------
@st.cache_data
def load_careers():
    df = pd.read_csv("OccupationData.csv")  # <-- Place your dataset here
    # Normalize columns
    df = df.rename(columns=lambda x: x.lower().strip())
    # If the dataset has 'title' and 'description' columns:
    if "title" in df.columns and "description" in df.columns:
        careers = df[["title", "description"]].dropna().head(200)  # use top 200 to save memory
    else:
        careers = df.head(200)
    return careers

careers_df = load_careers()

# Turn a sample of the dataset into a short text list
career_text = "\n".join(
    [f"{row['title']}: {row['description'][:120]}..." for _, row in careers_df.iterrows()]
)

# -----------------------------
# QUIZ FORM
# -----------------------------
with st.form("career_form"):
    st.subheader("🎯 Quick Career Quiz")
    interests = st.text_area("What topics or activities excite you most?")
    skills = st.text_area("What are your strongest skills?")
    dream_job = st.text_input("Describe your dream job or ideal lifestyle (optional):")
    submitted = st.form_submit_button("🚀 Get My Career Recommendations")

# -----------------------------
# AI CAREER ADVISOR
# -----------------------------
if submitted:
    with st.spinner("🤖 Matching you to real-world careers..."):
        prompt = f"""
You are an expert career advisor. Here are real careers from a dataset:

{career_text}

Based on this list and the user's info, choose the 3 careers that best fit them.

For each, include:
### 1. Career Name
**Why it fits:** Personalized explanation.
**Career Description:** Explain what they actually do.
**Learning Roadmap:** Key topics or free resources (no links).
**First Step:** A realistic first action.

User Info:
- Interests: {interests}
- Skills: {skills}
- Dream Job: {dream_job}
"""
        result = model(prompt, max_new_tokens=350, temperature=0.7)
        st.success("✅ Your Personalized AI Career Path is Ready!")
        st.markdown(result[0]["generated_text"])

st.markdown("---")
st.caption("Powered by LaMini-Flan-T5-248M")
