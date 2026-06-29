import streamlit as st
import requests
from PIL import Image

# ----------------------------
# API CONFIG
# ----------------------------
# Paste your FastAPI backend URL here.
FASTAPI_BASE_URL = "http://127.0.0.1:8000/predict"

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Fruit Freshness Detector",
    page_icon="🍎",
    layout="wide"
)

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.title {
    text-align: center;
    color: #2E8B57;
    font-size: 3rem;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 1.1rem;
    margin-bottom: 20px;
}

.result-box {
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------
st.markdown(
    '<div class="title">🍎 AI Fruit Freshness Detector 🍌</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Deep Learning CNN Model for Fresh vs Rotten Fruit Detection</div>',
    unsafe_allow_html=True
)

st.divider()

# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:

    st.header("📊 Model Information")

    st.success("Validation Accuracy: 98.17%")

    st.markdown("""
    ### 🧠 Model Details
    - CNN Architecture
    - TensorFlow / Keras
    - Binary Classification
    - Image Size: 224 × 224

    ### 🎯 Classes
    - Fresh
    - Rotten

    ### 📸 Supported Formats
    - JPG
    - JPEG
    - PNG
    """)

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a Fruit Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    # ----------------------------
    # CALL FASTAPI BACKEND
    # ----------------------------
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type,
        )
    }

    try:
        response = requests.post(FASTAPI_BASE_URL, files=files)
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Could not connect to the FastAPI backend. Make sure it's running.")
        st.stop()

    if response.status_code == 200:

        result = response.json()

        prediction = result["prediction"]      # "Fresh" or "Rotten"
        confidence = result["confidence"]       # float, percentage
        fresh_prob = result["fresh_probability"]    # float, percentage
        rotten_prob = result["rotten_probability"]  # float, percentage

        with col2:

            st.subheader("🔍 Prediction Result")

            if prediction == "Rotten":
                st.error("🍂 Rotten")
            else:
                st.success("🍏 Fresh")

            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )

        st.divider()

        # ----------------------------
        # PROBABILITY SECTION
        # ----------------------------
        st.subheader("📈 Prediction Probabilities")

        st.write(f"🍏 Fresh : {fresh_prob:.2f}%")
        st.progress(float(fresh_prob / 100))

        st.write(f"🍂 Rotten : {rotten_prob:.2f}%")
        st.progress(float(rotten_prob / 100))

        st.divider()

        # ----------------------------
        # FINAL VERDICT
        # ----------------------------
        st.subheader("🏆 Final Verdict")

        if prediction == "Rotten":
            st.error(
                f"⚠️ The fruit appears to be ROTTEN with {rotten_prob:.2f}% confidence."
            )
        else:
            st.success(
                f"✅ The fruit appears to be FRESH with {fresh_prob:.2f}% confidence."
            )

    else:
        st.error("Failed to get prediction from FastAPI backend.")

# ----------------------------
# FOOTER
# ----------------------------
st.divider()

st.markdown(
    """
    <div class="footer">
        🚀 Built with TensorFlow, Keras & Streamlit <br>
        CNN-Based Fruit Freshness Detection System
    </div>
    """,
    unsafe_allow_html=True
)