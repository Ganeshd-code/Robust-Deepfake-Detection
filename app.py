import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import time

from utils.transforms import (
    get_spatial_infer_transform,
    get_frequency_infer_transform
)

from utils.fft import get_fft

from models.spatial_model import SpatialModel
from models.frequency_model import FrequencyModel
from models.hybrid_model import HybridModel


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Deepfake Forensic System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background-color: #0e1117;
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.stMetric {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #2f3542;
}

div.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    background-color: #2563eb;
    color: white;
    border: none;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #1d4ed8;
}

img {
    border-radius: 12px;
    max-width: 100%;
}

video {
    max-width: 700px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# DEVICE
# =========================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.backends.cudnn.benchmark = True


# =========================================================
# HEADER
# =========================================================
st.title("🛡️ Robust Deepfake Detection System")

st.markdown("""
### Spatial • Frequency • Hybrid • Asymmetric Hybrid

AI-powered multimedia forensic analysis using:
- Spatial CNN learning
- FFT frequency analysis
- Hybrid feature fusion
- Robustness-aware asymmetric training
""")


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙️ System Information")

st.sidebar.success(f"Device: {device}")

if torch.cuda.is_available():

    st.sidebar.info(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

else:

    st.sidebar.warning(
        "Running on CPU"
    )

st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Select Detection Mode",
    [
        "Image Detection",
        "Video Detection"
    ]
)


# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():

    spatial = SpatialModel().to(device)

    spatial.load_state_dict(
        torch.load(
            "models/spatial.pth",
            map_location=device,
            weights_only=True
        )
    )

    spatial.eval()

    frequency = FrequencyModel().to(device)

    frequency.load_state_dict(
        torch.load(
            "models/frequency.pth",
            map_location=device,
            weights_only=True
        )
    )

    frequency.eval()

    hybrid = HybridModel().to(device)

    hybrid.load_state_dict(
        torch.load(
            "models/hybrid.pth",
            map_location=device,
            weights_only=True
        )
    )

    hybrid.eval()

    asymmetric = HybridModel().to(device)

    asymmetric.load_state_dict(
        torch.load(
            "models/asymmetric.pth",
            map_location=device,
            weights_only=True
        )
    )

    asymmetric.eval()

    return (
        spatial,
        frequency,
        hybrid,
        asymmetric
    )


(
    spatial_model,
    frequency_model,
    hybrid_model,
    asymmetric_model
) = load_models()


# =========================================================
# TRANSFORMS
# =========================================================
spatial_transform = (
    get_spatial_infer_transform()
)

frequency_transform = (
    get_frequency_infer_transform()
)


# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_image(image):

    fft_image = get_fft(image)

    rgb_tensor = torch.as_tensor(
        spatial_transform(image)
    ).unsqueeze(0).to(device)

    fft_tensor = torch.as_tensor(
        frequency_transform(fft_image)
    ).unsqueeze(0).to(device)

    outputs = {}

    with torch.no_grad():

        spatial_probs = torch.softmax(
            spatial_model(rgb_tensor),
            dim=1
        )[0]

        frequency_probs = torch.softmax(
            frequency_model(fft_tensor),
            dim=1
        )[0]

        hybrid_probs = torch.softmax(
            hybrid_model(
                rgb_tensor,
                fft_tensor
            ),
            dim=1
        )[0]

        asymmetric_probs = torch.softmax(
            asymmetric_model(
                rgb_tensor,
                fft_tensor
            ),
            dim=1
        )[0]

    outputs["Spatial"] = spatial_probs
    outputs["Frequency"] = frequency_probs
    outputs["Hybrid"] = hybrid_probs
    outputs["Asymmetric"] = asymmetric_probs

    return outputs, fft_image


# =========================================================
# DISPLAY RESULTS
# =========================================================
def display_results(outputs):

    st.subheader("📊 Model Predictions")

    cols = st.columns(4)

    fake_votes = 0
    suspicious_votes = 0

    for idx, (model_name, probs) in enumerate(
        outputs.items()
    ):

        real_prob = float(probs[0])
        fake_prob = float(probs[1])

        if model_name == "Asymmetric":

            if fake_prob >= 0.90:

                label = "FAKE"
                confidence = fake_prob

                fake_votes += 1

            elif fake_prob >= 0.70:

                label = "SUSPICIOUS"
                confidence = fake_prob

                suspicious_votes += 1

            else:

                label = "REAL"
                confidence = real_prob

        else:

            pred = int(
                torch.argmax(probs)
            )

            label = (
                "FAKE"
                if pred == 1
                else "REAL"
            )

            confidence = max(
                real_prob,
                fake_prob
            )

            if label == "FAKE":

                fake_votes += 1

        with cols[idx]:

            st.metric(
                model_name,
                label
            )

            st.progress(
                float(confidence)
            )

            st.write(
                f"Confidence: {confidence:.4f}"
            )

    st.markdown("---")

    st.subheader(
        "🧠 Final Forensic Analysis"
    )

    if fake_votes >= 3:

        st.error("""
        HIGH PROBABILITY OF
        MANIPULATION DETECTED
        """)

    elif fake_votes >= 1 or suspicious_votes >= 1:

        st.warning("""
        IMAGE APPEARS SUSPICIOUS
        """)

    else:

        st.success("""
        IMAGE APPEARS
        LIKELY AUTHENTIC
        """)


# =========================================================
# IMAGE DETECTION
# =========================================================
if mode == "Image Detection":

    uploaded_image = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_image.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:

            st.error(
                "Failed to load image."
            )

            st.stop()

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        outputs, fft_image = predict_image(
            image
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Original Image",
                width=400
            )

        with col2:

            st.image(
                fft_image,
                caption="FFT Frequency Representation",
                width=400
            )

        if st.button("Run Detection"):

            start_time = time.time()

            display_results(outputs)

            end_time = time.time()

            st.info(
                f"Inference Time: "
                f"{end_time-start_time:.3f} sec"
            )

            with st.expander(
                "Debug Probabilities"
            ):

                st.write(outputs)


# =========================================================
# VIDEO DETECTION
# =========================================================
if mode == "Video Detection":

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        tfile = tempfile.NamedTemporaryFile(
            delete=False
        )

        tfile.write(
            uploaded_video.read()
        )

        st.video(
            uploaded_video,
            start_time=0
        )

        if st.button(
            "Run Video Analysis"
        ):

            cap = cv2.VideoCapture(
                tfile.name
            )

            frame_count = 0

            sampled_frames = []

            spatial_scores = []
            frequency_scores = []
            hybrid_scores = []
            asymmetric_scores = []

            progress_bar = st.progress(0)

            st.info(
                "Extracting and analyzing frames..."
            )

            while cap.isOpened():

                ret, frame = cap.read()

                if (
                    not ret or
                    frame is None
                ):
                    break

                # SAMPLE EVERY 15TH FRAME
                if frame_count % 15 == 0:

                    frame_rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    outputs, _ = predict_image(
                        frame_rgb
                    )

                    spatial_scores.append(
                        float(outputs["Spatial"][1])
                    )

                    frequency_scores.append(
                        float(outputs["Frequency"][1])
                    )

                    hybrid_scores.append(
                        float(outputs["Hybrid"][1])
                    )

                    asymmetric_scores.append(
                        float(outputs["Asymmetric"][1])
                    )

                    if len(sampled_frames) < 6:

                        sampled_frames.append(
                            frame_rgb
                        )

                frame_count += 1

                progress_bar.progress(
                    min(
                        frame_count / 300,
                        1.0
                    )
                )

            cap.release()

            if len(hybrid_scores) == 0:

                st.error(
                    "No frames could be processed."
                )

            else:

                spatial_avg = float(
                    np.mean(spatial_scores)
                )

                frequency_avg = float(
                    np.mean(frequency_scores)
                )

                hybrid_avg = float(
                    np.mean(hybrid_scores)
                )

                asymmetric_avg = float(
                    np.mean(asymmetric_scores)
                )

                final_score = (
                    hybrid_avg +
                    asymmetric_avg
                ) / 2

                st.subheader(
                    "🎥 Video Forensic Analysis"
                )

                info1, info2, info3 = st.columns(3)

                with info1:

                    st.metric(
                        "Frames Analyzed",
                        len(hybrid_scores)
                    )

                with info2:

                    st.metric(
                        "Final Fake Score",
                        f"{final_score:.4f}"
                    )

                with info3:

                    if final_score >= 0.5:

                        st.metric(
                            "Final Verdict",
                            "FAKE"
                        )

                    else:

                        st.metric(
                            "Final Verdict",
                            "REAL"
                        )

                st.markdown("---")

                st.subheader(
                    "📊 Model-wise Video Predictions"
                )

                cols = st.columns(4)

                model_results = {
                    "Spatial": spatial_avg,
                    "Frequency": frequency_avg,
                    "Hybrid": hybrid_avg,
                    "Asymmetric": asymmetric_avg
                }

                fake_votes = 0

                for idx, (
                    model_name,
                    score
                ) in enumerate(
                    model_results.items()
                ):

                    if score >= 0.5:

                        label = "FAKE"

                        fake_votes += 1

                    else:

                        label = "REAL"

                    with cols[idx]:

                        st.metric(
                            model_name,
                            label
                        )

                        st.progress(
                            float(score)
                        )

                        st.write(
                            f"Fake Score: {score:.4f}"
                        )

                st.markdown("---")

                st.subheader(
                    "🧠 Final Forensic Report"
                )

                if fake_votes >= 3:

                    st.error("""
                    HIGH PROBABILITY OF
                    VIDEO MANIPULATION
                    """)

                elif fake_votes >= 1:

                    st.warning("""
                    VIDEO APPEARS SUSPICIOUS
                    """)

                else:

                    st.success("""
                    VIDEO APPEARS
                    LIKELY AUTHENTIC
                    """)

                st.markdown("---")

                st.subheader(
                    "🖼️ Sampled Video Frames"
                )

                frame_cols = st.columns(3)

                for idx, frame in enumerate(
                    sampled_frames
                ):

                    with frame_cols[idx % 3]:

                        st.image(
                            frame,
                            width=250
                        )

                with st.expander(
                    "Detailed Video Scores"
                ):

                    st.write({
                        "Spatial": spatial_avg,
                        "Frequency": frequency_avg,
                        "Hybrid": hybrid_avg,
                        "Asymmetric": asymmetric_avg,
                        "Final Ensemble": final_score
                    })