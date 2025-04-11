import streamlit as st
import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer
import base64

# Load the model and vectorizer
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Define class names
class_names = ['Fake', 'Real']

# Set up Streamlit page
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detector with Explainability (LIME)")

# Input text area
user_input = st.text_area("Enter news article content to check if it's Fake or Real:")

# Function for LIME explainability
def explain_prediction(text):
    explainer = LimeTextExplainer(class_names=class_names)

    def predict_proba_fn(texts):
        tfidf = vectorizer.transform(texts)
        return model.predict_proba(tfidf)

    exp = explainer.explain_instance(
        text,
        predict_proba_fn,
        num_features=8,
        labels=[0, 1]
    )
    return exp.as_html()

# On button click, predict and explain
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some news content first.")
    else:
        # Predict
        transformed_text = vectorizer.transform([user_input])
        prediction = model.predict(transformed_text)[0]
        probability = model.predict_proba(transformed_text)[0][prediction]

        label = "🟥 FAKE NEWS" if prediction == 0 else "🟩 REAL NEWS"
        st.markdown(f"### 🧠 Prediction: {label}")
        st.markdown(f"#### Confidence: {probability * 100:.2f}%")

        # Explain prediction with LIME
        st.markdown("### 🔍 Explanation (LIME):")
        html_exp = explain_prediction(user_input)
        st.components.v1.html(html_exp, height=800, scrolling=True)
