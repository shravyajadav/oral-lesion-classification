import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Oral Lesion Classification",
    page_icon="🦷",
    layout="wide"
)

# ---------------- WEBSITE DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.header {
    background: linear-gradient(120deg, #123c69, #2874a6);
    padding: 30px 20px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 25px;
}

.header h1 {
    color: white;
    font-size: 34px;
}

.header p {
    color: #eaf4ff;
    font-size: 17px;
}

.section-title {
    color: #123c69;
    font-size: 23px;
    font-weight: bold;
}

.result-card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #d8e3f0;
}

div.stButton > button {
    background-color: #1769aa;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #123c69;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PROJECT TITLE ----------------
st.markdown("""
<div class="header">
    <h1>🦷 Oral Lesion Classification</h1>
    <p>Using Image Processing and Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("models/tuned_lightgbm_pipeline.pkl")

try:
    model = load_model()
except Exception as error:
    st.error("Unable to load the trained model.")
    st.caption(str(error))
    st.stop()

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(image):
    image = np.array(image.convert("RGB"))
    image = cv2.resize(image, (128, 128))

    features = []

    # RGB colour features: 6
    for channel in range(3):
        features.append(np.mean(image[:, :, channel]))
        features.append(np.std(image[:, :, channel]))

    # Grayscale statistical features: 3
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    features.extend([
        np.mean(gray),
        np.std(gray),
        np.var(gray)
    ])

    # GLCM texture features: 6
    gray_quantized = (gray // 16).astype(np.uint8)

    glcm = graycomatrix(
        gray_quantized,
        distances=[1],
        angles=[0],
        levels=16,
        symmetric=True,
        normed=True
    )

    for prop in [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM"
    ]:
        features.append(graycoprops(glcm, prop)[0, 0])

    return np.array(features).reshape(1, -1)


# ---------------- IMAGE UPLOAD ----------------
st.markdown(
    '<p class="section-title">📤 Upload an Oral Image</p>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image from your computer",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Upload an image to preview it and obtain a prediction.")

else:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns(2, gap="large")

        # ---------------- IMAGE PREVIEW ----------------
        with left:
            st.markdown(
                '<p class="section-title">🖼️ Image Preview</p>',
                unsafe_allow_html=True
            )

            st.image(
                image,
                caption="Uploaded Oral Image",
                use_container_width=True
            )

        # ---------------- PREDICTION ----------------
        with right:
            st.markdown(
                '<p class="section-title">🔍 Prediction Result</p>',
                unsafe_allow_html=True
            )

            st.write("Click below to classify the uploaded image.")

            if st.button("Analyse Image", use_container_width=True):
                try:
                    with st.spinner("Analysing image..."):
                        features = extract_features(image)

                        raw_prediction = model.predict(features)[0]

                        # Assumed training label mapping:
                        # 0 = Normal
                        # 1 = Malignant
                        label_mapping = {
                            "0": "Normal",
                            "1": "Malignant"
                        }

                        predicted_class = label_mapping.get(
                            str(raw_prediction),
                            f"Unknown class ({raw_prediction})"
                        )

                        st.session_state["prediction"] = predicted_class
                        st.session_state["image_name"] = uploaded_file.name

                        if hasattr(model, "predict_proba"):
                            probabilities = model.predict_proba(features)[0]
                            confidence = float(np.max(probabilities)) * 100
                        else:
                            confidence = None

                        st.session_state["confidence"] = confidence

                except Exception as error:
                    st.error("Prediction failed.")
                    st.caption(str(error))

            # Keep the result visible after reruns
            if (
                st.session_state.get("image_name") == uploaded_file.name
                and "prediction" in st.session_state
            ):
                st.markdown(
                    '<div class="result-card">',
                    unsafe_allow_html=True
                )

                st.markdown("### Predicted Class")
                st.markdown(
                    f"## {st.session_state['prediction']}"
                )

                confidence = st.session_state.get("confidence")

                if confidence is not None:
                    st.metric(
                        "Model Prediction Confidence",
                        f"{confidence:.1f}%"
                    )

                st.markdown("</div>", unsafe_allow_html=True)

    except Exception as error:
        st.error("Unable to process the uploaded image.")
        st.caption(str(error))

# ---------------- FOOTER ----------------
st.divider()

st.caption(
    "Educational research prototype only. Predictions are not a "
    "medical diagnosis. Consult a qualified healthcare professional "
    "for assessment of oral lesions."
)
