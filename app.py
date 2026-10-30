import streamlit as st
import numpy as np
import cv2
import joblib
from skimage.feature import graycomatrix, graycoprops

MODEL_PATH = "models/tuned_lightgbm_pipeline.pkl"

st.set_page_config(
    page_title="Oral Lesion Image Classification",
    page_icon="🔬",
    layout="centered"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def extract_features(image):
    image = cv2.resize(image, (128, 128))

    # Convert BGR to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    features = []

    # RGB channel means and standard deviations: 6 features
    for channel in range(3):
        features.append(float(np.mean(rgb[:, :, channel])))
        features.append(float(np.std(rgb[:, :, channel])))

    # Grayscale statistical features: 3 features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features.append(float(np.mean(gray)))
    features.append(float(np.std(gray)))
    features.append(float(np.var(gray)))

    # GLCM texture features: 6 features
    quantized = (gray // 16).astype(np.uint8)

    glcm = graycomatrix(
        quantized,
        distances=[1],
        angles=[0],
        levels=16,
        symmetric=True,
        normed=True
    )

    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM"
    ]

    for prop in properties:
        features.append(
            float(graycoprops(glcm, prop)[0, 0])
        )

    return np.array(features, dtype=float).reshape(1, -1)


st.title("🔬 Oral Lesion Image Classification")

st.write(
    "Upload an oral image to obtain an experimental "
    "Normal or Malignant classification."
)

st.warning(
    "Research prototype only. This tool cannot diagnose "
    "oral cancer or replace professional medical assessment."
)

uploaded_file = st.file_uploader(
    "Upload an oral image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.frombuffer(
        uploaded_file.getvalue(),
        dtype=np.uint8
    )

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not read this image. Please upload another.")
    else:
        preview = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(
            preview,
            caption="Uploaded oral image",
            use_container_width=True
        )

        if st.button("Predict", type="primary"):
            try:
                model = load_model()
                features = extract_features(image)

                prediction = model.predict(features)[0]

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features)[0]
                    confidence = float(np.max(probabilities)) * 100
                else:
                    confidence = None

                if isinstance(prediction, (int, np.integer)):
                    label = {
                        0: "Normal",
                        1: "Malignant"
                    }.get(int(prediction), str(prediction))
                else:
                    label = str(prediction)

                st.subheader("Prediction Result")
                st.write(f"**Predicted class:** {label}")

                if confidence is not None:
                    st.write(
                        f"**Model confidence:** {confidence:.2f}%"
                    )

                st.caption(
                    "The displayed probability is a model output, "
                    "not a measure of clinical certainty."
                )

            except Exception as error:
                st.error(
                    "Prediction failed. Check that the model and "
                    "feature-extraction method match."
                )
                st.exception(error)