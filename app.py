import streamlit as st
import time
import cv2
import tempfile
import os
import numpy as np

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Video Check",
    page_icon="🎥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =========================================================
# SESSION STATE
# =========================================================

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "show_results" not in st.session_state:
    st.session_state.show_results = False

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# =========================================================
# COLORS / THEME
# =========================================================

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

# =========================================================
# CSS
# =========================================================

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

h1, h2, h3, h4, p, label {{
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

# =========================================================
# VIDEO ANALYSIS FUNCTION
# =========================================================

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            "success": False,
            "message": "The video could not be opened or decoded."
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 0.0

    duration = frame_count / fps if fps > 0 else 0.0

    if frame_count <= 0:
        cap.release()
        return {
            "success": False,
            "message": "No readable frames were found in this video."
        }

    sample_count = min(16, frame_count)

    sample_indices = np.linspace(
        0,
        frame_count - 1,
        sample_count,
        dtype=int
    )

    frames = []
    sharpness_values = []
    brightness_values = []

    for frame_index in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()

        if not success or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        small_gray = cv2.resize(gray, (64, 64))

        sharpness = float(
            cv2.Laplacian(gray, cv2.CV_64F).var()
        )

        brightness = float(np.mean(gray))

        frames.append(small_gray)
        sharpness_values.append(sharpness)
        brightness_values.append(brightness)

    cap.release()

    if len(frames) < 2:
        return {
            "success": False,
            "message": "The video did not contain enough readable frames."
        }

    differences = []

    for index in range(1, len(frames)):
        difference = float(
            np.mean(
                cv2.absdiff(
                    frames[index - 1],
                    frames[index]
                )
            )
        )

        differences.append(difference)

    average_difference = float(np.mean(differences))
    average_sharpness = float(np.mean(sharpness_values))

    duplicate_threshold = 1.5

    duplicate_count = sum(
        1 for value in differences
        if value < duplicate_threshold
    )

    duplicate_ratio = duplicate_count / max(1, len(differences))

    observations = []

    if width < 360 or height < 240:
        observations.append("Low-resolution video")

    if fps > 0 and fps < 15:
        observations.append("Low frame rate")

    if average_sharpness < 20:
        observations.append("Frames appear soft or blurry")

    if duplicate_ratio > 0.45:
        observations.append("Many sampled frames are visually similar")

    if average_difference < 2.5:
        observations.append("Very little visual movement was detected")

    if not observations:
        observations.append("No major technical irregularities detected")

    if duplicate_ratio > 0.65:
        assessment = "TECHNICALLY UNUSUAL"
        explanation = (
            "The sampled video contains a high amount of repeated "
            "visual content. This can occur because of editing, "
            "still-image animation, screen recording, or processing."
        )

    elif average_difference < 1.5:
        assessment = "LIMITED VISUAL EVIDENCE"
        explanation = (
            "The sampled frames show very little visual change. "
            "The system cannot make a meaningful authenticity assessment."
        )

    else:
        assessment = "NO OBVIOUS TECHNICAL ANOMALY"
        explanation = (
            "The video showed readable frames and normal visual "
            "variation. This does not prove that the video is authentic."
        )

    return {
        "success": True,
        "assessment": assessment,
        "explanation": explanation,
        "duration": duration,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "frames_analyzed": len(frames),
        "sharpness": average_sharpness,
        "duplicate_ratio": duplicate_ratio,
        "observations": observations
    }

# =========================================================
# TOP HEADER
# =========================================================

top_left, top_right = st.columns([5, 1])

with top_left:
    st.markdown(
        "<h1 style='font-size:42px; margin-bottom:0;'>AI Video Check</h1>",
        unsafe_allow_html=True
    )

with top_right:
    st.write("")

    theme_button = "☀️" if st.session_state.theme == "dark" else "🌙"

    if st.button(theme_button, help="Toggle dark/light mode"):
        if st.session_state.theme == "dark":
            st.session_state.theme = "light"
        else:
            st.session_state.theme = "dark"

        st.rerun()

st.markdown(
    "<div class='subtitle'>Check whether a video may contain AI-generated or manipulated content.</div>",
    unsafe_allow_html=True
)

# =========================================================
# RESULTS PAGE
# =========================================================

if st.session_state.show_results:

    result = st.session_state.analysis_result

    st.markdown(
        "<div class='completed'>ANALYSIS COMPLETED</div>",
        unsafe_allow_html=True
    )

    if result is None or not result.get("success", False):

        error_message = "Analysis failed."

        if result is not None:
            error_message = result.get(
                "message",
                "Analysis failed."
            )

        st.error(error_message)

    else:

        st.markdown(
            f"""
<div class='result-card'>
    <h2 style='text-align:center;'>Analysis Results</h2>
    <h3 style='text-align:center; margin-top:20px;'>
        {result["assessment"]}
    </h3>
    <p style='text-align:center; color:{secondary};'>
        {result["explanation"]}
    </p>
</div>
""",
            unsafe_allow_html=True
        )

        st.subheader("Technical Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Video Duration",
                f"{result['duration']:.2f} seconds"
            )

            st.metric(
                "Resolution",
                f"{result['width']} × {result['height']}"
            )

            st.metric(
                "Frames Analyzed",
                result["frames_analyzed"]
            )

        with col2:
            st.metric(
                "Frame Rate",
                f"{result['fps']:.2f} FPS"
            )

            st.metric(
                "Average Sharpness",
                f"{result['sharpness']:.1f}"
            )

            st.metric(
                "Repeated Frame Ratio",
                f"{result['duplicate_ratio'] * 100:.1f}%"
            )

        st.subheader("Observations")

        for observation in result["observations"]:
            st.write("• " + observation)

        st.info(
            "This is a technical screening tool, not definitive proof "
            "of whether a video is authentic or AI-generated."
        )

    st.write("")

    if st.button("✕ Close Results", use_container_width=True):
        st.session_state.show_results = False
        st.session_state.analysis_result = None
        st.rerun()

# =========================================================
# MAIN PAGE
# =========================================================

else:

    st.markdown(
        "<div class='section-title'>Paste video link</div>",
        unsafe_allow_html=True
    )

    video_link = st.text_input(
        "Video URL",
        placeholder="Paste a public video link here...",
        label_visibility="collapsed"
    )

    st.markdown(
        "<div style='text-align:center; color:#888; margin:22px 0;'>OR</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='section-title'>Upload video</div>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=[
            "mp4",
            "mov",
            "avi",
            "mkv",
            "webm"
        ],
        label_visibility="collapsed"
    )

    st.write("")

    analyze_button = st.button(
        "Analyze Video",
        use_container_width=True,
        type="primary"
    )

    if analyze_button:

        if uploaded_file is None:

            if video_link:
                st.warning(
                    "Video-link analysis is not enabled yet. "
                    "Please upload the video file directly."
                )
            else:
                st.error(
                    "Please upload a video file first."
                )

        else:

            progress_bar = st.progress(0)
            status_text = st.empty()

            stages = [
                (10, "Preparing video..."),
                (25, "Loading video data..."),
                (
