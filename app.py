import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os

# Title
st.title("STEMPath – AI Career & Learning Guide")
st.write("Discover your ideal STEM career based on your interests and strengths.")

# Load or embed dataset
@st.cache_data
def load_data():
    df = pd.read_csv("OccupationData.csv")
    if 'title' not in df.columns or 'description' not in df.columns:
        st.error("Dataset must include 'title' and 'description' columns.")
        st.stop()
    return df

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def embed_dataset(df):
    model = load_model()
    return model.encode(df['description'].tolist(), convert_to_tensor=True)

# Load everything
df = load_data()
model = load_model()
embeddings = embed_dataset(df)

# Quiz UI
st.subheader("Career Quiz")
interests = st.text_area("What topics or activities excite you most?")
skills = st.text_area("What are your strongest skills?")
dream = st.text_area("Describe your dream job or ideal life (optional)")

# Predict careers
if st.button("Find Careers"):
    if not interests and not skills and not dream:
        st.warning("Please enter some details first.")
    else:
        with st.spinner("Analyzing your profile..."):
            user_input = " ".join([interests, skills, dream])
            user_emb = model.encode(user_input, convert_to_tensor=True)

            # Compute similarities
            scores = util.cos_sim(user_emb, embeddings)[0]
            top_indices = torch.topk(scores, k=3).indices

            st.subheader("Top 3 Career Matches")
            for i, idx in enumerate(top_indices):
                st.markdown(f"**{i+1}. {df.iloc[idx]['title']}**")
                st.caption(f"Similarity: {scores[idx]:.3f}")

st.markdown("---")
st.markdown("🔬 *Powered by open-source AI – No API required*")
