import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import time

st.set_page_config(
    page_title="AI Video Check",
    layout="centered"
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "results" not in st.session_state:
    st.session_state.results = None

dark = st.session_state.theme == "dark"

background = "#0d0f14" if dark else "#f5f6f8"
text = "#f1f1f1" if dark else "#17191f"
card = "#191c24" if dark else "#ffffff"
border = "#30343f" if dark else "#d9dce3"
secondary = "#a8abb4" if dark else "#60636c"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {background};
        color: {text};
    }}
    h1, h2, h3, p, label {{
        color: {text} !important;
    }}
    .subtitle {{
        color: {secondary};
        font-size: 17px;
        margin-bottom: 30px;
    }}
    .card {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 25px;
        margin-top: 20px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

header_left, header_right = st.columns([5, 1])

with header_left:
    st.title("AI Video Check")

with header_right:
    if st.button("☀️" if dark else "🌙"):
        st.session_state.theme = "light" if dark else "dark"
        st.rerun()

st.markdown(
    '<div class="subtitle">Check whether a video may contain AI-generated or manipulated content.</div>',
    unsafe_allow_html=True
)


def analyze_video(path):
    capture = cv2.VideoCapture(path)

    if not capture.isOpened():
        return None, "The video could not be opened."

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 0.0

    duration = total_frames / fps if fps > 0 else 0.0

    if total_frames <= 0:
        capture.release()
        return None, "No readable frames were found."

    sample_count = min(12, total_frames)

    indices = np.linspace(
        0,
        total_frames - 1,
        sample_count,
        dtype=int
    )

    frames = []
    sharpness = []

    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = capture.read()

        if not success or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))

        frames.append(resized)

        value = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        sharpness.append(float(value))

    capture.release()

    if len(frames) < 2:
        return None, "Not enough readable frames were found."

    differences = []

    for i in range(1, len(frames)):
        difference = np.mean(
            cv2.absdiff(
                frames[i - 1],
                frames[i]
            )
        )
        differences.append(float(difference))

    average_difference = float(np.mean(differences))
    average_sharpness = float(np.mean(sharpness))

    repeated = sum(
        value < 1.5
        for value in differences
    )

    repeated_ratio = repeated / max(1, len(differences))

    observations = []

    if width < 360 or height < 240:
        observations.append("Low video resolution")

    if fps > 0 and fps < 15:
        observations.append("Low frame rate")

    if average_sharpness < 20:
        observations.append("Frames appear blurry")

    if repeated_ratio > 0.45:
        observations.append("Many sampled frames are visually similar")

    if average_difference < 2.5:
        observations.append("Very little visual movement detected")

    if not observations:
        observations.append("No major technical irregularities detected")

    if repeated_ratio > 0.65:
        assessment = "TECHNICALLY UNUSUAL"
        explanation = "A high amount of repeated visual content was detected."

    elif average_difference < 1.5:
        assessment = "LIMITED VISUAL EVIDENCE"
        explanation = "The video showed very little visual change."

    else:
        assessment = "NO OBVIOUS TECHNICAL ANOMALY"
        explanation = "The video showed readable frames and normal visual variation."

    result = {
        "assessment": assessment,
        "explanation": explanation,
        "duration": duration,
        "fps": fps,
        "width": width,
        "height": height,
        "frames": len(frames),
        "sharpness": average_sharpness,
        "repeated": repeated_ratio,
        "observations": observations
    }

    return result, None


if st.session_state.results is None:

    st.subheader("Paste video link")

    video_link = st.text_input(
        "Video URL",
        placeholder="Paste a public video link here...",
        label_visibility="collapsed"
    )

    st.markdown(
        '<p style="text-align:center;">OR</p>',
        unsafe_allow_html=True
    )

    st.subheader("Upload video")

    uploaded = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        label_visibility="collapsed"
    )

    analyze_button = st.button(
        "Analyze Video",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        if uploaded is None:

            if video_link:
                st.warning(
                    "Link analysis is not enabled yet. Please upload the video file."
                )
            else:
                st.error("Please upload a video file first.")

        else:

            progress = st.progress(0)
            status = st.empty()

            messages = [
                (15, "Preparing video..."),
                (35, "Reading video data..."),
                (55, "Extracting frames..."),
                (75, "Checking frame consistency..."),
                (90, "Preparing report..."),
                (100, "Analysis complete.")
            ]

            for percentage, message in messages:
                time.sleep(0.2)
                progress.progress(percentage)
                status.write(f"{message} {percentage}%")

            extension = os.path.splitext(uploaded.name)[1]
            temporary_path = None

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=extension
                ) as temporary_file:

                    temporary_file.write(uploaded.getbuffer())
                    temporary_path = temporary_file.name

                result, error = analyze_video(temporary_path)

                if error:
                    st.error(error)
                else:
                    st.session_state.results = result
                    st.rerun()

            except Exception as error:
                st.error("Analysis failed: " + str(error))

            finally:

                if (
                    temporary_path is not None
                    and os.path.exists(temporary_path)
                ):
                    os.remove(temporary_path)

else:

    result = st.session_state.results

    st.success("ANALYSIS COMPLETED")

    st.markdown(
        f"""
        <div class="card">
        <h2>{result["assessment"]}</h2>
        <p>{result["explanation"]}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Technical Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Duration", f'{result["duration"]:.2f} seconds')
        st.metric(
            "Resolution",
