import streamlit as st
import joblib
from lime.lime_text import LimeTextExplainer

model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
explainer = LimeTextExplainer(class_names=["Fake", "Real"])

st.title("📰 Fake News Detection with LIME")
text_input = st.text_area("Enter news text:")

if st.button("Predict"):
    if text_input:
        transformed_text = vectorizer.transform([text_input])
        prediction = model.predict(transformed_text)[0]
        st.write("### Prediction:", "🟥 Fake" if prediction == 0 else "🟩 Real")

        st.write("### LIME Explanation:")
        exp = explainer.explain_instance(text_input, model.predict_proba, num_features=6, labels=[0, 1],
                                         vectorizer=vectorizer.transform)
        st.components.v1.html(exp.as_html(), height=800)
