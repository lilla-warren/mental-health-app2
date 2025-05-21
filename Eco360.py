# Eco360 – Smart Spatial Regret Engine + Design DNA + Eco Suggestions
mental-health-app2/
Eco360.py
├── requirements.txt
├── .streamlit/
│   └── secrets.toml

import streamlit as st
import openai
import pandas as pd

# --- API Key ---
openai.api_key = st.secrets["openai"]["api_key"]
[openai]
api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# --- App Config ---
st.set_page_config(page_title="Eco360 – Spatial Insight AI", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .stTextInput > div > div > input { color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 Eco360: Regret Risk & Design DNA")
st.markdown("Understand how your space will make you *feel* — now and in the future.")

# --- Lifestyle Questions for Regret Risk Engine ---
st.subheader("🔍 Tell us about your habits and design mindset")
with st.form("eco360_form"):
    q1 = st.selectbox("Do you tend to rearrange or change furniture often?", ["Yes", "No", "Sometimes"])
    q2 = st.selectbox("Do you prefer minimalist or maximalist environments?", ["Minimalist", "Maximalist", "Balanced"])
    q3 = st.selectbox("Are you sensitive to clutter or visual noise?", ["Very", "A little", "Not really"])
    q4 = st.selectbox("Do you expect major life changes (kids, moving, work-from-home)?", ["Yes", "Maybe", "No"])
    q5 = st.selectbox("Do you tend to regret aesthetic decisions over time?", ["Often", "Rarely", "Never"])
    q6 = st.selectbox("How sustainable or eco-friendly are your choices?", ["Very", "Moderate", "Not usually"])
    submitted = st.form_submit_button("🧠 Analyze My Design")

if submitted:
    # --- Combine answers into GPT prompt ---
    lifestyle_answers = f"""
    - Rearranging: {q1}
    - Style: {q2}
    - Clutter sensitivity: {q3}
    - Life changes expected: {q4}
    - Regret tendency: {q5}
    - Sustainability: {q6}
    """

    prompt = f"""
    A user provided the following lifestyle answers regarding their space design:
    {lifestyle_answers}

    Based on this, generate the following:

    1. Regret Risk Report:
       - Predict potential issues with space design based on the user's personality and lifestyle.
       - Include risks like fatigue, mismatch, future stress, or impracticality.

    2. Design DNA Card:
       - Summarize their spatial personality and values.
       - Include traits like adaptability, clutter sensitivity, energy orientation, and sustainability score.

    3. Eco-Friendly Furniture Suggestions:
       - Suggest specific types of furniture, materials, or modular options that align with the user's Design DNA.
       - Consider future flexibility, environmental impact, and emotional fit.
    """

    with st.spinner("Generating insight using GPT-4..."):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an empathetic, future-aware spatial design expert and psychologist with sustainability expertise."},
                {"role": "user", "content": prompt}
            ]
        )
        gpt_output = response['choices'][0]['message']['content']

    st.subheader("📉 Regret Risk Report + 🧬 Design DNA + ♻️ Smart Eco Suggestions")
    st.markdown(gpt_output)

st.caption("Eco360 by you – Powered by intention, emotion, and sustainable AI.")
