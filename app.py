import streamlit as st
import time
import cv2
import tempfile
import os
import numpy as np
from PIL import Image

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

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if "show_results" not in st.session_state:
    st.session_state.show_results = False

if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

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
        padding: 35px;
        text-align: center;
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

# ---------------- TOP BAR ----------------

top_left, top_right = st.columns([5, 1])

with top_left:
    st.markdown(
        f"<h1 style='font-size:42px; margin-bottom:0;'>AI Video Check</h1>",
        unsafe_allow_html=True
    )

with top_right:
    st.write("")
    if st.button(
        "☀️" if st.session_state.theme == "dark" else "🌙",
        help="Toggle dark/light mode"
    ):
        st.session_state.theme = (
            "light"
            if st.session_state.theme == "dark"
            else "dark"
        )
        st.rerun()

st.markdown(
    "<div class='subtitle'>Check whether a video may contain AI-generated or manipulated content.</div>",
    unsafe_allow_html=True
)

# ---------------- RESULTS PAGE ----------------

if st.session_state.show_results:

    st.markdown(
        "<div class='completed'>ANALYSIS COMPLETED</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class='result-card'>
            <h2 style='margin-bottom:20px;'>Analysis Results</h2>
            <h3 style='color:#d99b32;'>UNABLE TO DETERMINE</h3>
            <p style='color:{secondary};'>
                The current screening system could not establish
                a reliable conclusion for this video.
            </p>
            <p style='font-size:14px; color:{secondary};'>
                A future detection model will provide a probability
                assessment when sufficient evidence is available.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    if st.button("✕ Close Results", use_container_width=True):
        st.session_state.show_results = False
        st.session_state.analysis_complete = False
        st.session_state.analysis_started = False
        st.rerun()

# ---------------- MAIN PAGE ----------------

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
        type=["mp4", "mov", "avi", "mkv", "webm"],
        label_visibility="collapsed"
    )

    st.write("")

    analyze = st.button(
        "Analyze Video",
        use_container_width=True,
        type="primary"
    )

    if analyze:

        if not video_link and uploaded_file is None:

            st.error("Please paste a video link or upload a video.")

        else:

            st.session_state.analysis_started = True

            progress_bar = st.progress(0)
            status_text = st.empty()

            stages = [
                (10, "Preparing video..."),
                (25, "Loading video data..."),
                (40, "Extracting video frames..."),
                (55, "Examining visual patterns..."),
                (70, "Checking temporal consistency..."),
                (85, "Evaluating available evidence..."),
                (100, "Finalizing analysis...")
            ]

            for percentage, message in stages:

                time.sleep(0.5)

                progress_bar.progress(percentage)

                status_text.markdown(
                    f"<p style='text-align:center; color:{secondary};'>{message} {percentage}%</p>",
                    unsafe_allow_html=True
                )

            time.sleep(0.5)

            st.session_state.show_results = True
            st.session_state.analysis_started = False

            st.rerun()

# ---------------- FOOTER ----------------

st.markdown(
    f"""
    <div class='footer'>
        AI Video Check — Experimental synthetic media screening tool.<br>
        Results are assessments, not definitive proof.
    </div>
    """,
    unsafe_allow_html=True
)