# Streamlit Mental Health Diagnostic Assistant (Enhanced with Real Data + GPT-4)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import openai # type: ignore

# --- API Key ---
openai.api_key = st.secrets["openai"]["api_key"]

# --- Load and preprocess real dataset ---
df = pd.read_csv("https://raw.githubusercontent.com/lilla-warren/mental-health-app2/main/survey.csv")
columns_of_interest = [
    'Age', 'Gender', 'self_employed', 'family_history', 'treatment',
    'work_interfere', 'no_employees', 'remote_work', 'tech_company',
    'benefits', 'care_options', 'wellness_program', 'mental_health_consequence'
]

df_clean = df[columns_of_interest].dropna()

# Encode categorical variables
label_encoders = {}
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        label_encoders[col] = le

X = df_clean.drop(columns=['treatment'])
y = LabelEncoder().fit_transform(df_clean['treatment'])

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# --- Streamlit UI ---
st.set_page_config(page_title="🧠 Smart Mental Health Assistant", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .stTextInput > div > div > input { color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Mental Health Diagnostic Assistant (Real Data + GPT-4)")
st.markdown("Answer the questions below or type your symptoms for analysis.")

# --- Form for ML prediction ---
st.subheader("📋 Answer These to Get Prediction")
with st.form("ml_form"):
    user_input = {}
    for col in X.columns:
        val = st.selectbox(f"{col.replace('_', ' ').capitalize()}", options=label_encoders[col].classes_)
        user_input[col] = label_encoders[col].transform([val])[0]
    submitted = st.form_submit_button("Get Prediction")

if submitted:
    prediction = model.predict([list(user_input.values())])[0]
    treatment_prediction = "Yes" if prediction == 1 else "No"
    st.success(f"🧾 Prediction: Likely to need treatment? **{treatment_prediction}**")

# --- GPT Chat Style ---
user_symptoms = st.chat_input("🗨️ Or describe how you're feeling:")
if user_symptoms:
    with st.spinner("Analyzing with GPT-4..."):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a compassionate and intelligent mental health assistant."},
                {"role": "user", "content": f"These are the symptoms: {user_symptoms}. Analyze this, suggest possible concerns, and offer insight."}
            ]
        )
        gpt_message = response['choices'][0]['message']['content']
    st.chat_message("assistant").markdown(gpt_message)

st.caption("This tool is for educational purposes only. Always consult a licensed professional for mental health advice.")
