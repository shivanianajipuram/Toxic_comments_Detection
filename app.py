import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# App title
st.title("🛡️ Toxic Comment Detection App")

st.write("Detect whether a comment is Toxic or Non-Toxic")

# User input
comment = st.text_area("Enter a Comment")

# Prediction
if st.button("Analyze Comment"):

    if comment.strip() == "":
        st.warning("Please enter a comment")
    else:
        cleaned = clean_text(comment)

        vectorized = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized)[0]

        probability = model.predict_proba(vectorized)[0][1]

        # Progress bar
        st.progress(int(probability * 100))

        st.write(f"### Toxicity Score: {probability * 100:.2f}%")

        # Result
        if prediction == 1:
            st.error("⚠️ Toxic Comment Detected")
        else:
            st.success("✅ Non-Toxic Comment")