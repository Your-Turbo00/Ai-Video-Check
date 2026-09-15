import streamlit as st
import time
import cv2
import tempfile
import os
import numpy as np

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="AI Video Check",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------- SESSION STATE ----------------

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "show_results" not in st.session_state:
    st.session_state.show_results = False

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# ---------------- THEME ----------------

if st.session_state.theme == "dark":
    background = "#0d0f14"
    text = "#f1f1f1"
    secondary = "#a8abb4"
    card = "#191c24"
    border = "#30343f"
    input_bg = "#151820"
else:
    background = "#f5f6f8"
    text = "#17191f"
    secondary = "#60636c"
    card = "#ffffff"
    border = "#d9dce3"
    input_bg = "#ffffff"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {background};
        color: {text};
    }}

    .main .block-container {{
        max-width: 850px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }}

    h1, h2, h3, p, label {{
        color: {text} !important;
    }}

    .subtitle {{
        color: {secondary};
        font-size: 17px;
        margin-top: -12px;
        margin-bottom: 35px;
    }}

    .section-title {{
        color: {text};
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 10px;
    }}

    .result-card {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 30px;
        margin-top: 25px;
    }}

    .completed {{
        font-size: 30px;
        font-weight: 700;
        text-align: center;
        margin-top: 35px;
        margin-bottom: 20px;
        color: {text};
    }}

    .stTextInput input {{
        background-color: {input_bg} !important;
        color: {text} !important;
        border: 1px solid {border} !important;
    }}

    .stTextInput input::placeholder {{
        color: {secondary} !important;
    }}

    div[data-testid="stFileUploader"] {{
        background-color: {card};
        border-radius: 12px;
        padding: 8px;
    }}

    .metric-box {{
        background-color: {input_bg};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
        margin-bottom: 10px;
    }}

    .footer {{
        color: {secondary};
        text-align: center;
        font-size: 13px;
        margin-top: 60px;
        padding-top: 20px;
        border-top: 1px solid {border};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- VIDEO ANALYSIS ----------------

def analyze_video(video_path):
    """
    Performs technical forensic screening.

    This is not a definitive AI/deepfake classifier.
    It checks:
    - Video readability
    - Duration
    - FPS
    - Resolution
    - Frame sharpness
    - Frame duplication
    - Visual variation between frames
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            "status": "error",
            "message": "The video could not be opened or decoded."
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps is None or fps <= 0:
        fps = 0

    duration = frame_count / fps if fps > 0 else 0

    # Limit analysis to prevent Streamlit Cloud resource overload
    max_samples = 16

    if frame_count > 0:
        sample_indices = np.linspace(
            0,
            frame_count - 1,
            min(max_samples, frame_count),
            dtype=int
        )
    else:
        sample_indices = []

    frames = []
    sharpness_values = []
    brightness_values = []

    for index in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = cap.read()

        if not success or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Laplacian variance is a common sharpness indicator
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        brightness = float(np.mean(gray))

        # Resize for lightweight comparison
        small_frame = cv2.resize(gray, (64, 64))

        frames.append(small_frame)
        sharpness_values.append(sharpness)
        brightness_values.append(brightness)

    cap.release()

    if len(frames) < 2:
        return {
            "status": "error",
            "message": "Not enough readable frames were found."
        }

    # Calculate differences between consecutive sampled frames
    frame_differences = []

    for i in range(1, len(frames)):
        difference = np.mean(
            cv2.absdiff(frames[i - 1], frames[i])
        )
        frame_differences.append(float(difference))

    average_difference = float(np.mean(frame_differences))
    average_sharpness = float(np.mean(sharpness_values))
    average_brightness = float(np.mean(brightness_values))

    # Duplicate-frame indicator
    duplicate_threshold = 1.5
    duplicate_count = sum(
        1 for difference in frame_differences
        if difference < duplicate_threshold
    )

    duplicate_ratio = duplicate_count / max(1, len(frame_differences))

    # Basic technical observations
    observations = []

    if width < 360 or height < 240:
        observations.append("Low-resolution video")

    if fps > 0 and fps < 15:
        observations.append("Low frame rate")

    if average_sharpness < 20:
        observations.append("Very soft or blurry frames")

    if duplicate_ratio > 0.45:
        observations.append("High amount of repeated visual frames")

    if average_difference < 2.5:
        observations.append("Very little visual movement between sampled frames")

    if not observations:
        observations.append("No major technical irregularities detected")

    # Conservative screening interpretation
    if duplicate_ratio > 0.65:
        assessment = "TECHNICALLY UNUSUAL"
        explanation = (
            "The video contains a high proportion of visually repeated "
            "frames. This may result from editing, still-image animation, "
            "screen recording, or other processing."
        )
    elif average_difference < 1.5:
        assessment = "LIMITED VISUAL EVIDENCE"
        explanation = (
            "The sampled frames show very little visual change, so the "
            "system cannot make a meaningful authenticity assessment."
        )
    else:
        assessment = "NO OBVIOUS TECHNICAL ANOMALY"
        explanation = (
            "The sampled frames were readable and showed normal visual "
            "variation. This does not prove that the video is authentic."
        )

    return {
        "status": "success",
        "assessment": assessment,
        "explanation": explanation,
        "duration": duration,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "frames_analyzed": len(frames),
        "sharpness": average_sharpness,
        "frame_difference": average_difference,
        "duplicate_ratio": duplicate_ratio,
        "observations": observations
    }


# ---------------- TOP BAR ----------------

top_left, top_right = st.columns([5, 1])

with top_left:
    st.markdown(
        "<h1 style='font-size:42px; margin-bottom:0
