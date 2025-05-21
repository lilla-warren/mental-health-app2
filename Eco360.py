# Eco360 – ML-Powered Regret Risk Predictor + Design DNA Engine (No OpenAI)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Config ---
st.set_page_config(page_title="Eco360 – ML Spatial Predictor", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .stTextInput > div > div > input { color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 Eco360: ML-Driven Spatial Regret Predictor")
st.markdown("Discover layout risks and long-term satisfaction using real-time data and pattern recognition.")

# --- Sample Dataset (simulating survey responses) ---
data = {
    'rearranges_frequently': [1, 0, 0, 1, 0, 1],
    'prefers_style': ['minimalist', 'maximalist', 'balanced', 'minimalist', 'balanced', 'maximalist'],
    'clutter_sensitivity': ['high', 'low', 'medium', 'high', 'medium', 'low'],
    'life_changes_expected': [1, 0, 1, 0, 1, 1],
    'regret_decisions': ['often', 'rarely', 'never', 'often', 'rarely', 'often'],
    'eco_friendly': ['high', 'medium', 'low', 'medium', 'high', 'low'],
    'regret_risk': ['high', 'low', 'low', 'high', 'medium', 'high']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode categorical variables
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Split features and target
X = df.drop(columns=['regret_risk'])
y = df['regret_risk']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ML Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- User Form Input ---
st.subheader("📝 Tell us about your preferences")
with st.form("user_input_form"):
    rearranges = st.selectbox("Do you rearrange or change your space often?", ["Yes", "No"])
    style = st.selectbox("Your preferred design style?", ["minimalist", "maximalist", "balanced"])
    clutter = st.selectbox("How sensitive are you to visual clutter?", ["high", "medium", "low"])
    changes = st.selectbox("Expect any big life changes soon?", ["Yes", "No"])
    regret = st.selectbox("How often do you regret design decisions?", ["often", "rarely", "never"])
    eco = st.selectbox("How sustainable are your furniture choices?", ["high", "medium", "low"])
    form_submitted = st.form_submit_button("🔍 Predict Regret Risk")

if form_submitted:
    # Encode inputs
    input_data = {
        'rearranges_frequently': 1 if rearranges == "Yes" else 0,
        'prefers_style': label_encoders['prefers_style'].transform([style])[0],
        'clutter_sensitivity': label_encoders['clutter_sensitivity'].transform([clutter])[0],
        'life_changes_expected': 1 if changes == "Yes" else 0,
        'regret_decisions': label_encoders['regret_decisions'].transform([regret])[0],
        'eco_friendly': label_encoders['eco_friendly'].transform([eco])[0]
    }

    prediction = model.predict([list(input_data.values())])[0]
    result_label = label_encoders['regret_risk'].inverse_transform([prediction])[0]

    st.success(f"🔮 Your Regret Risk Level is: **{result_label.upper()}**")

    # Live Recommendations
    st.subheader("📦 Live Recommendations")
    if result_label == 'high':
        st.warning("🛋️ Consider more flexible, modular furniture that can evolve with your needs.")
        st.warning("🧠 Try decluttering and reducing visual noise for mental relief.")
    elif result_label == 'medium':
        st.info("📐 Balance trend and function. Choose items with lasting appeal.")
        st.info("🌿 Include 1–2 signature sustainable pieces.")
    else:
        st.success("✅ Your layout habits seem low-risk! Maintain flexibility and update as your values evolve.")

# --- Visual Analytics ---
st.subheader("📊 Dataset Insights")
fig, ax = plt.subplots()
sns.countplot(x=label_encoders['regret_risk'].inverse_transform(y), palette="coolwarm", ax=ax)
ax.set_title("Distribution of Regret Risk Ratings")
st.pyplot(fig)

st.caption("Eco360 – Empowering conscious living through data and design.")
