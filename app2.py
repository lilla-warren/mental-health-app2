import streamlit as st  # type: ignore
import numpy as np  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
import pandas as pd  # type: ignore

# Define categorized symptoms
symptom_categories = {
    "Cognitive": ["hallucinations", "delusions", "memory_loss", "paranoia"],
    "Emotional": ["anxiety", "depressed_mood"],
    "Behavioral": ["impulsivity", "sleep_disturbance"]
}

# All symptoms
all_symptoms = [symptom for group in symptom_categories.values() for symptom in group]

# Dummy dataset for training
data = {
    "hallucinations": [1, 0, 0, 0, 0],
    "delusions": [1, 1, 0, 0, 0],
    "anxiety": [0, 0, 1, 0, 1],
    "depressed_mood": [0, 0, 1, 1, 0],
    "impulsivity": [0, 0, 0, 1, 0],
    "memory_loss": [0, 0, 0, 0, 1],
    "sleep_disturbance": [0, 0, 1, 1, 1],
    "paranoia": [1, 1, 0, 0, 1],
    "diagnosis": ["schizophrenia", "bipolar_disorder", "ptsd", "major_depression", "borderline_personality"]
}

df = pd.DataFrame(data)
X = df[all_symptoms]
y = df["diagnosis"]

model = RandomForestClassifier()
model.fit(X, y)

# Diagnosis descriptions
descriptions = {
    "schizophrenia": "Characterized by hallucinations, delusions, and disorganized thinking.",
    "bipolar_disorder": "Marked by episodes of mania and depression.",
    "ptsd": "Triggered by traumatic events, with flashbacks and severe anxiety.",
    "major_depression": "Persistent feelings of sadness, hopelessness, and loss of interest.",
    "borderline_personality": "Emotional instability, impulsive behavior, and unstable relationships."
}

# Streamlit UI
st.set_page_config(page_title="Mental Health Diagnostic Assistant", page_icon="🧠")

# Custom background
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://www.transparenttextures.com/patterns/connected.png");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Mental Health Diagnostic Assistant")
st.write("Select your symptoms below to receive a probability-based diagnosis suggestion.")

# Symptom category filter
category = st.selectbox("Choose a symptom category", list(symptom_categories.keys()))
selected_symptoms = st.multiselect("Select symptoms", symptom_categories[category])

# Predict
if selected_symptoms:
    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]
    proba = model.predict_proba([input_vector])[0]
    labels = model.classes_

    # Top result
    top_index = np.argmax(proba)
    top_label = labels[top_index]
    top_prob = proba[top_index]

    st.subheader("📊 Diagnosis Probabilities")
    for label, prob in zip(labels, proba):
        st.write(f"**{label}**: {prob:.2f}")

    st.success(f"🧾 Most Likely Diagnosis: **{top_label}** ({top_prob:.2f})")
    st.info(descriptions.get(top_label, "No description available."))

    # Alert if low confidence
    if top_prob < 0.5:
        st.warning("⚠️ Results are inconclusive. Please consult a mental health professional.")
else:
    st.info("Please select at least one symptom.")

st.caption("Note: This tool is a basic prototype and not a substitute for professional evaluation.")
