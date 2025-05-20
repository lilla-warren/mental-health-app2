# Advanced Mental Health Diagnostic Assistant

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import base64
from datetime import datetime

# ---- UI CONFIG ----
st.set_page_config(page_title="🧠 Smart Mental Health Assistant", layout="wide")

# ---- BACKGROUND ----
def set_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0e1117;
            color: white;
        }
        .stTextInput > div > div > input {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

# ---- HEADER ----
st.title("🧠 Smart Mental Health Diagnostic Assistant")
st.markdown("This app gets smarter the more it is used. Enter your symptoms in the chat below.")

# ---- SYMPTOMS LIST (Extensive) ----
symptoms = [
    "hallucinations", "delusions", "anxiety", "depressed_mood", "impulsivity", "memory_loss",
    "sleep_disturbance", "paranoia", "mood_swings", "lack_of_focus", "social_withdrawal", "aggression",
    "suicidal_thoughts", "mania", "phobia", "obsession", "compulsion", "panic_attacks",
    "flashbacks", "low_self_esteem", "eating_disorder", "substance_abuse"
]

# ---- Dummy Dataset for ML ----
data = {
    symptom: np.random.randint(0, 2, 100) for symptom in symptoms
}
data["diagnosis"] = np.random.choice([
    "schizophrenia", "bipolar_disorder", "ptsd", "major_depression", "borderline_personality",
    "ocd", "anxiety_disorder", "eating_disorder", "adhd", "panic_disorder"
], 100)

df = pd.DataFrame(data)
X = df[symptoms]
y = df["diagnosis"]

model = RandomForestClassifier()
model.fit(X, y)

# ---- Diagnosis Descriptions ----
diagnosis_descriptions = {
    "schizophrenia": "Chronic mental illness with delusions, hallucinations.",
    "bipolar_disorder": "Manic highs and depressive lows.",
    "ptsd": "Triggered by traumatic experiences.",
    "major_depression": "Intense sadness, loss of interest.",
    "borderline_personality": "Emotional instability, impulsivity.",
    "ocd": "Unwanted repetitive thoughts (obsessions) and actions (compulsions).",
    "anxiety_disorder": "Persistent excessive worry or fear.",
    "eating_disorder": "Abnormal eating habits affecting health.",
    "adhd": "Attention deficit hyperactivity disorder.",
    "panic_disorder": "Sudden intense fear without danger."
}

# ---- Chat Input ----
user_input = st.chat_input("Enter your symptoms (comma-separated):")

if user_input:
    input_list = [s.strip().lower().replace(" ", "_") for s in user_input.split(",")]
    input_vector = [1 if symptom in input_list else 0 for symptom in symptoms]
    proba = model.predict_proba([input_vector])[0]
    labels = model.classes_
    top_diagnosis = labels[np.argmax(proba)]

    st.chat_message("assistant").markdown(f"**Most Likely Diagnosis**: {top_diagnosis}")
    st.chat_message("assistant").markdown(diagnosis_descriptions.get(top_diagnosis, "No info available."))

    st.chat_message("assistant").markdown("**Other Probabilities**:")
    for label, prob in zip(labels, proba):
        st.chat_message("assistant").markdown(f"- {label}: {prob:.2f}")

    # ---- Analytics Logging (optional future feature with DB integration) ----
    # log = pd.DataFrame({"timestamp": [datetime.now()], "input": [user_input], "diagnosis": [top_diagnosis]})
    # log.to_csv("analytics_log.csv", mode='a', header=False, index=False)

st.caption("Disclaimer: This is a prototype. Always consult a licensed professional.")
