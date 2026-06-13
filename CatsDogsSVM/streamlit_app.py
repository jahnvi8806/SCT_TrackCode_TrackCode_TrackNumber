import streamlit as st
import joblib
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="AI Pet Classifier",
    page_icon="🐶",
    layout="wide"
)

# CSS
st.markdown("""
<style>
.main{
    background-color:#0E1117;
}
.hero{
    background:linear-gradient(135deg,#6C63FF,#3B82F6);
    padding:30px;
    border-radius:20px;
    text-align:center;
    color:white;
}
.result{
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class='hero'>
<h1>🐱🐶 AI Cats vs Dogs Classifier</h1>
<h4>Machine Learning Powered Image Recognition</h4>
</div>
""", unsafe_allow_html=True)

st.write("")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=120)
    st.title("Model Info")
    st.success("Algorithm: SVM")
    st.info("Feature Extraction: PCA")
    st.warning("Binary Classification")

# Layout
col1, col2 = st.columns([1.3,1])

with col1:

    uploaded_file = st.file_uploader(
        "📤 Upload Cat/Dog Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

with col2:

    st.subheader("📊 Prediction Dashboard")

    if uploaded_file:

        if st.button("🚀 Analyze Image"):

            # Dummy Output
            label = "🐶 Dog"
            confidence = 82

            st.markdown(
                f"<div class='result'>{label}</div>",
                unsafe_allow_html=True
            )

            st.progress(confidence)

            st.metric(
                label="Confidence",
                value=f"{confidence}%"
            )

            st.subheader("Probability")

            st.write("🐱 Cat")
            st.progress(18)

            st.write("🐶 Dog")
            st.progress(82)

            if confidence > 80:
                st.success("High Confidence Prediction")
            elif confidence > 60:
                st.warning("Moderate Confidence Prediction")
            else:
                st.error("Low Confidence Prediction")

st.divider()

c1,c2,c3 = st.columns(3)

c1.metric("Classes", "2")
c2.metric("Model", "SVM")
c3.metric("Feature Size", "64x64")

st.markdown(
"""
---
<center>
Made with ❤️ using Streamlit | SkillCraft Technology Internship
</center>
""",
unsafe_allow_html=True
)