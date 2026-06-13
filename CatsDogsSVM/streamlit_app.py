import streamlit as st
import joblib
import cv2
import numpy as np
from PIL import Image

# Load trained model
import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "svm_cats_dogs_model.joblib"
)

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐶",
    layout="centered"
)

st.title("🐱🐶 Cats vs Dogs Classifier")
st.write("Upload an image and let the SVM model predict.")

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features_from_image(image):
    img = np.array(image)

    # RGB -> BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Same preprocessing as training
    img = cv2.resize(img, (64, 64))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    pixel_features = gray.flatten()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()

    color_features = np.concatenate([hist_h, hist_s, hist_v])
    color_features = color_features / (color_features.sum() + 1e-7)

    features = np.concatenate([pixel_features, color_features])

    return features.astype(np.float32)

# -----------------------------
# Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Cat/Dog Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        features = extract_features_from_image(image)

        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]

        cat_prob = float(probabilities[0])
        dog_prob = float(probabilities[1])

        if prediction == 0:
            label = "🐱 Cat"
            confidence = cat_prob
        else:
            label = "🐶 Dog"
            confidence = dog_prob

        st.success(f"Prediction: {label}")

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.write("### Probability")

        st.write(f"🐱 Cat: {cat_prob * 100:.2f}%")
        st.progress(int(cat_prob * 100))

        st.write(f"🐶 Dog: {dog_prob * 100:.2f}%")
        st.progress(int(dog_prob * 100))