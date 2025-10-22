import streamlit as st
from transformers import pipeline

# Cache model to avoid reloading
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

model = load_model()

st.set_page_config(page_title="STEMPath – AI Career & Learning Guide", page_icon="🧠", layout="centered")

st.title("🧠 STEMPath – AI Career & Learning Guide")
st.markdown("Find your ideal career path powered by open-source AI — no API keys, no limits!")

st.divider()

# --- Step 1: Quiz ---
with st.form("career_form"):
    st.subheader("🎯 Quick Career Quiz")
    interests = st.text_area("What topics or activities excite you most? (e.g. robotics, coding, biology, space, art, etc.)")
    skills = st.text_area("What are your strongest skills? (e.g. creativity, problem-solving, teamwork, math)")
    dream_job = st.text_input("Describe your dream job or what kind of life you'd like (optional):")

    submitted = st.form_submit_button("✨ Get My AI Career Recommendation")

# --- Step 2: Run model ---
if submitted:
    with st.spinner("🤖 Thinking hard about your future..."):
        prompt = f"""
You are an expert AI career advisor. Based on the user's interests, skills, and dream job, suggest the 3 most fitting careers in the entire world (not just STEM). 
For each career, include:

1. **Career Name**
2. **Why this fits the user**
3. **Career Description**
4. **Learning Roadmap** (with free resources like YouTube, MOOCs, or open projects — no links, just ideas)
5. **First Step to Start Today**

Be detailed, friendly, and realistic.

User Info:
- Interests: {interests}
- Skills: {skills}
- Dream Job: {dream_job}
"""
        result = model(prompt, max_new_tokens=500, temperature=0.8)
        response = result[0]["generated_text"]

    st.success("✅ Your Personalized AI Career Path is Ready!")
    st.markdown(response)

    st.markdown("---")
    st.caption("Powered by FLAN-T5-Base – Runs 100% locally, no API required.")
