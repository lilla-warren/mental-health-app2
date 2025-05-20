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
# Symptom categories
symptom_categories = {
    "Cognitive": ["hallucinations", "delusions", "memory_loss", "paranoia"],
    "Emotional": ["anxiety", "depressed_mood"],
    "Behavioral": ["impulsivity", "sleep_disturbance"]
}

selected_symptoms = []

st.subheader("🔍 Select your symptoms by category")
for category, symptoms in symptom_categories.items():
    with st.expander(f"{category} Symptoms"):
        selected = st.multiselect(f"Select {category.lower()} symptoms", symptoms)
        selected_symptoms.extend(selected)
diagnosis_descriptions = {
    "schizophrenia": "A chronic mental disorder involving hallucinations, delusions, and disordered thinking.",
    "bipolar_disorder": "A disorder causing extreme mood swings including emotional highs (mania) and lows (depression).",
    "ptsd": "Post-traumatic stress def set_bg_hack(main_bg):
    main_bg_ext = "png"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

import base64
set_bg_hack("your_image.png")  # Make sure this image is in your project directory
disorder: a condition triggered by a terrifying event.",
    "major_depression": "A mood disorder causing a persistent feeling of sadness and loss of interest.",
    "borderline_personality": "A disorder affecting emotions and self-image, leading to unstable relationships and mood."
}
top_diagnosis = labels[np.argmax(proba)]
st.success(f"🧠 Most Likely Diagnosis: {top_diagnosis}")
st.write(diagnosis_descriptions.get(top_diagnosis, "Description not available."))
max_prob = max(proba)
if max_prob < 0.5:
    st.warning("⚠️ The prediction confidence is low. Please consult a professional.")
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
df = pd.read_csv("C:/Users/97155/Downloads/archive (1)/survey.csv")
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
