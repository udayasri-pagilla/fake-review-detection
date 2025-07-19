import streamlit as st
import joblib
import base64

# Load model
model = joblib.load('model/fake_genuine_model.joblib')

# Streamlit UI with custom background and styled layout
st.set_page_config(page_title="Fake vs Genuine Review Classifier", layout="centered")

# Set background color and padding with custom CSS
st.markdown("""
    <style>
    body {
        background-color: #f5f5f5;
    }
    .reportview-container {
        background: #f0f2f6;
        padding: 2rem;
    }
    .stTextArea textarea {
        background-color: #ffffff;
        border-radius: 10px;
        font-size: 16px;
        padding: 12px;
    }
    .stButton>button {
        background-color: #6c63ff;
        color: white;
        font-weight: bold;
        padding: 10px 24px;
        border-radius: 8px;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background-color: #5145cd;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align: center; color: #6c63ff;'>🛡️ Fake vs Genuine Review Classifier</h1>
<p style='text-align: center;'>Enter a product review below to find out if it's real or fake.</p>
""", unsafe_allow_html=True)

user_input = st.text_area("📝 Your Review:", height=150)

if st.button("Classify Review"):
    if user_input.strip():
        prediction = model.predict([user_input])[0]
        if prediction == "genuine":
            st.success("✅ This review is classified as **Genuine**.")
        else:
            st.error("🚫 This review is classified as **Fake**.")
    else:
        st.warning("⚠️ Please enter a review first.")
