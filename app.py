# app.py
import streamlit as st
from transformers import pipeline

import streamlit as st
from transformers import pipeline

# Cache model so it doesn’t reload every time
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-small")

model = load_model()

st.set_page_config(page_title="STEMPath – AI Career & Learning Guide", page_icon="🧠", layout="centered")

st.title("🧠 STEMPath – AI Career & Learning Guide")
st.markdown("Discover your ideal STEM career path and a personalized learning roadmap powered by AI!")

st.divider()

# --- Step 1: Quiz Form ---
with st.form("career_quiz"):
    st.subheader("🎯 Quick Quiz")
    interests = st.text_area("What topics or activities excite you most? (e.g. coding, biology, design)")
    skills = st.text_area("What are your strongest skills? (e.g. math, creativity, problem-solving)")
    dream_job = st.text_input("Describe your dream job in one sentence (optional):")

    submitted = st.form_submit_button("Get My AI Career Path")

if submitted:
    with st.spinner("Analyzing your responses..."):
        # Build the prompt for AI model
        prompt = (
            f"You are a career guidance assistant. Based on the user's interests and skills, "
            f"suggest 3 possible STEM careers and a short learning roadmap for each. "
            f"Interests: {interests}\nSkills: {skills}\nDream Job: {dream_job}\n\n"
            f"Format your response as:\n"
            f"**Career 1:** ...\nLearning Path: ...\n\n"
            f"**Career 2:** ...\nLearning Path: ...\n\n"
            f"**Career 3:** ...\nLearning Path: ..."
        )

        result = model(prompt, max_new_tokens=300, temperature=0.7)
        response = result[0]["generated_text"]

    st.success("✅ Here's your personalized STEM career guide!")
    st.markdown(response)

    st.markdown("---")
    st.caption("Powered by google/flan-t5-small (runs locally, no external API).")


# Load the local Hugging Face model (offline)
@st.cache_resource
def load_model():
    # You can switch to "google/flan-t5-base" for stronger results
    return pipeline("text2text-generation", model="google/flan-t5-small")

model = load_model()

# Basic descriptions and resources for common STEM fields
CAREERS = {
    "data science": {
        "desc": "Analyze data, build models, and discover insights using statistics and AI.",
        "resources": [
            "Kaggle Learn: Intro to Machine Learning",
            "Google Data Analytics Certificate",
            "freeCodeCamp Data Science Course",
        ],
        "projects": [
            "Titanic survival model",
            "Sales forecast predictor",
            "AI-powered dashboard",
        ],
    },
    "software engineering": {
        "desc": "Design and develop reliable software, apps, and systems.",
        "resources": [
            "CS50 by Harvard",
            "The Odin Project (Full Stack)",
            "Python & JavaScript crash courses",
        ],
        "projects": [
            "Portfolio website",
            "Chat app with Flask or Node.js",
            "REST API backend",
        ],
    },
    "robotics": {
        "desc": "Combine hardware, coding, and AI to build intelligent robotic systems.",
        "resources": [
            "Intro to Robotics (Coursera)",
            "ROS tutorials",
            "MIT OpenCourseWare: Control Systems",
        ],
        "projects": [
            "Line-following robot",
            "AI robot arm with computer vision",
        ],
    },
    "bioengineering": {
        "desc": "Apply engineering principles to solve biological and medical challenges.",
        "resources": [
            "CrashCourse Biology",
            "MIT Bioengineering Basics",
            "Coursera Biomedical Engineering Specialization",
        ],
        "projects": [
            "Simulate a prosthetic limb system",
            "Design a biosensor concept",
        ],
    },
    "cybersecurity": {
        "desc": "Protect digital systems and networks from vulnerabilities and attacks.",
        "resources": [
            "TryHackMe beginner path",
            "CompTIA Security+ learning guide",
            "HackTheBox labs",
        ],
        "projects": [
            "Set up a honeypot VM",
            "Password manager app",
        ],
    },
    "ux design": {
        "desc": "Design user-centered interfaces and research to improve user experiences.",
        "resources": [
            "Google UX Design Certificate",
            "NNgroup UX Tutorials",
        ],
        "projects": [
            "Redesign an app onboarding flow",
            "Usability test for a web app",
        ],
    },
}

# Streamlit UI
st.set_page_config(page_title="STEMPath – AI Career Predictor", layout="centered")
st.title("🧭 STEMPath – AI Career & Learning Guide (Offline AI)")
st.markdown("Discover your best STEM path based on your interests and skills — no API, runs locally!")

# Quiz form
with st.form("quiz"):
    interests = st.text_area("✨ What topics or subjects interest you most?")
    skills = st.text_area("🧩 What are your strongest skills or things you enjoy doing?")
    dream_job = st.text_input("🚀 Describe your dream job (in a few words):")
    submitted = st.form_submit_button("Predict My Career")

if submitted:
    if not interests or not skills or not dream_job:
        st.error("Please fill out all fields.")
    else:
        user_input = (
            f"Based on the following details, predict the best STEM career field:\n"
            f"Interests: {interests}\n"
            f"Skills: {skills}\n"
            f"Dream job: {dream_job}\n\n"
            f"Respond with only one field name, like: data science, robotics, bioengineering, cybersecurity, software engineering, or UX design."
        )

        st.info("Analyzing your profile with local AI model...")
        prediction = model(user_input, max_length=50)[0]["generated_text"].lower()
        st.success(f"🧠 Predicted Field: **{prediction.strip().capitalize()}**")

        # Match to known careers
        best_match = None
        for field in CAREERS.keys():
            if field in prediction:
                best_match = field
                break

        if best_match:
            career = CAREERS[best_match]
            st.subheader(f"🌟 Career: {best_match.title()}")
            st.write(career["desc"])

            # Generate learning roadmap
            roadmap_prompt = (
                f"Create a 5-step beginner-friendly learning roadmap for someone starting in {best_match}. "
                f"Include what to learn first, key skills, and practical projects."
            )
            roadmap = model(roadmap_prompt, max_length=220, temperature=0.7)[0]["generated_text"]

            st.markdown("### 🧭 Personalized Learning Roadmap")
            st.write(roadmap)

            st.markdown("### 📚 Recommended Resources")
            for res in career["resources"]:
                st.write(f"- {res}")

            st.markdown("### 💡 Project Ideas")
            for p in career["projects"]:
                st.write(f"- {p}")
        else:
            st.warning("Could not match predicted field to known categories. Try refining your answers!")

st.markdown("---")
st.caption("Powered locally by Hugging Face Transformers (FLAN-T5) — No API keys required ✅")
