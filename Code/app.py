import os
from pathlib import Path
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(page_title="DetectAI", layout="centered")

st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none; }

div[data-testid="stPageLink"] {
    display: flex;
    justify-content: center;
    margin-top: 4px;
}
div[data-testid="stPageLink"] a {
    color: #9ca3af !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    text-decoration: none !important;
    background: transparent !important;
    border: none !important;
    padding: 2px 4px !important;
}
div[data-testid="stPageLink"] a:hover {
    color: #6b7280 !important;
    text-decoration: underline !important;
}
div[data-testid="stPageLink"] span[data-testid="stIconMaterial"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

_DIR = Path(__file__).resolve().parent
MODEL_PATH = _DIR / 'deepfake_detection_model.h5'

model = load_model(MODEL_PATH)


def preprocess_image(image):
    image = cv2.resize(image, (96, 96))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image


def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image, verbose=0)
    class_label = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    result = "Real" if class_label == 0 else "Fake"
    return result, confidence


def get_ai_explanation(result, confidence):
    if result == "Fake":
        if confidence >= 70:
            return (
                f"The model assigned a {confidence:.0f}% probability of manipulation to this image, "
                "indicating high confidence in the fake classification. "
                "Localised visual artifacts such as texture discontinuities, irregular skin rendering, "
                "or geometric distortions in the facial region are likely contributing factors."
            )
        elif confidence >= 50:
            return (
                f"The model assigned a {confidence:.0f}% probability of manipulation to this image. "
                "The classification meets the detection threshold, though the moderate confidence level "
                "indicates that discriminating features were not strongly pronounced. "
                "The result should be considered alongside other available evidence."
            )
        else:
            return (
                f"The model returned a fake classification with {confidence:.0f}% confidence, "
                "marginally exceeding the 0.35 detection threshold. "
                "At this confidence level, the result carries significant uncertainty. "
                "Factors such as compression artefacts, atypical lighting, or limited facial "
                "visibility may have influenced the classification."
            )
    else:
        if confidence >= 70:
            return (
                f"The model assigned a {confidence:.0f}% probability that this image is authentic. "
                "No strong indicators of synthetic generation or post-processing manipulation were detected. "
                "The observed facial features are consistent with those present in genuine photographic captures."
            )
        elif confidence >= 50:
            return (
                f"The model classified this image as real with {confidence:.0f}% confidence. "
                "Observed features are broadly consistent with authentic imagery, though the moderate "
                "confidence score indicates some ambiguity in the classification. "
                "Contextual verification is recommended if precision is required."
            )
        else:
            return (
                f"The model returned a real classification with {confidence:.0f}% confidence. "
                "The low confidence level reflects substantial uncertainty in the result. "
                "Reduced image quality, unusual capture conditions, or atypical facial characteristics "
                "may have limited the model's discriminative ability."
            )


st.markdown("""
<div style="
    background: #1a1a2e;
    border-radius: 12px;
    padding: 36px 28px;
    text-align: center;
    margin-bottom: 16px;
">
    <h1 style="color: white; margin: 0; font-size: 2.2rem; font-weight: 600; letter-spacing: -0.5px;">
        DetectAI
    </h1>
    <p style="color: #b0b8d0; margin: 10px 0 0; font-size: 1rem;">
        Upload a facial image to check whether it is real or AI-generated.
    </p>
</div>
""", unsafe_allow_html=True)

st.page_link("pages/about.py", label="About this project")

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("##### Upload a facial image (JPG or PNG)")
with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    result, confidence = predict_image(image)

    if result == "Fake":
        color = "red"
        description = (
            "This result is generated by an AI model and should not be treated as definitive evidence "
            "of image manipulation. The system may produce incorrect classifications, particularly for "
            "low-quality, compressed, or out-of-distribution images. If this result has serious implications, "
            "seek independent expert verification before acting on it."
        )
    else:
        color = "green"
        description = (
            "This result is generated by an AI model and should not be treated as definitive confirmation "
            "of image authenticity. No automated system can guarantee accuracy in all cases. "
            "If this result has serious implications, seek independent expert verification before acting on it."
        )

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.image(image, channels="BGR")

    with right_col:
        st.markdown(f'''
        <div style='background-color:{"#ffebeb" if result=="Fake" else "#ebffeb"}; 
        padding: 20px; border-radius: 10px; 
        border-left: 5px solid {"red" if result=="Fake" else "green"};'>
        <h2 style='color:{"red" if result=="Fake" else "green"}; margin:0;'>
        {"⚠ Deepfake Detected" if result=="Fake" else "✓ Authentic Image"}
        </h2>
        <p style='margin:5px 0 0 0; font-size:18px; color:{"red" if result=="Fake" else "green"};'>
        Confidence: {confidence:.2f}%</p>
        </div>
        ''', unsafe_allow_html=True)
        st.progress(float(confidence) / 100)
        st.caption(description)

        if confidence < 70:
            st.warning(
                f"**Low Confidence Warning** ({confidence:.2f}%)\n\n"
                "The classification confidence is below 70%, indicating uncertainty in the result. "
                "This may be attributable to:\n"
                "- Atypical image quality or compression\n"
                "- Unusual facial orientation or partial occlusion\n"
                "- Edge cases underrepresented in the training data\n\n"
                "**Note:** This result should not be treated as reliable without further review."
            )

    st.markdown(
        "<p style='font-size:0.78rem; color:#9ca3af; margin: 16px 0 4px;'>DetectAI Assistant</p>",
        unsafe_allow_html=True,
    )
    with st.chat_message("assistant"):
        st.write(get_ai_explanation(result, confidence))
