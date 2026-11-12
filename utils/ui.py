"""모든 페이지에서 함께 사용하는 화면 스타일."""

import streamlit as st


def apply_app_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1480px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stAppViewContainer"] {
            background: #f6f8fb;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5eaf1;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 2px 8px rgba(20, 40, 70, 0.035);
        }
        [data-testid="stMetricLabel"] p {
            color: #5b6878;
            font-weight: 600;
        }
        [data-testid="stMetricValue"] {
            color: #18324d;
        }
        div.stButton > button, div.stDownloadButton > button {
            border-radius: 10px;
            font-weight: 600;
            min-height: 2.7rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 14px;
            border-color: #e5eaf1;
            background: #ffffff;
        }
        .app-eyebrow {
            color: #4d7194;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }
        .app-hero {
            background: linear-gradient(120deg, #18324d 0%, #2e5c7e 100%);
            color: white;
            padding: 2rem 2.1rem;
            border-radius: 18px;
            margin: 0.6rem 0 1.4rem 0;
        }
        .app-hero h2 {
            color: white;
            font-size: clamp(1.35rem, 2.3vw, 2rem);
            line-height: 1.45;
            margin: 0.25rem 0 0.5rem 0;
        }
        .app-hero p {
            color: #e4edf5;
            margin: 0;
        }
        .small-note {
            color: #627083;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
