import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Weight Predictor",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap" rel="stylesheet">

    <style>
    :root {
        --ink: #0B1F33;
        --paper: #EFF3F6;
        --surface: #FFFFFF;
        --line: #D9E0E7;
        --muted: #5B6B7C;
        --amber: #E8A317;
    }
    .stApp { background: var(--paper); }
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 720px; }

    body, .stApp, .stApp input, .stApp button, .stApp div[data-testid] {
        font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
        color: var(--ink);
    }

    .eyebrow {
        font-family: "IBM Plex Mono", monospace;
        font-size: .78rem;
        letter-spacing: .18em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: .4rem;
    }
    .eyebrow b { color: var(--amber); font-weight: 600; }

    .hero h1 {
        font-family: "Space Grotesk", sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: var(--ink);
        margin: 0 0 .45rem 0;
        line-height: 1.05;
    }
    .hero p {
        color: var(--muted);
        font-size: 1.05rem;
        margin: 0;
        max-width: 34rem;
    }

    .ruler {
        height: 26px;
        margin: 1.4rem 0 1.8rem 0;
        background:
            repeating-linear-gradient(to right, var(--amber) 0 2px, transparent 2px 100px) bottom left / 100% 26px no-repeat,
            repeating-linear-gradient(to right, rgba(11,31,51,.45) 0 1px, transparent 1px 10px) bottom left / 100% 9px no-repeat,
            repeating-linear-gradient(to right, rgba(11,31,51,.55) 0 1.5px, transparent 1.5px 50px) bottom left / 100% 17px no-repeat;
    }

    div[data-testid="stNumberInput"] label {
        color: var(--ink) !important;
        font-weight: 600;
        font-size: 1rem;
    }
    div[data-testid="stNumberInput"] input {
        background: var(--surface) !important;
        color: var(--ink) !important;
        border: 2px solid var(--line) !important;
        border-radius: 12px !important;
        font-family: "IBM Plex Mono", monospace !important;
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        height: 3.1rem;
    }
    div[data-testid="stNumberInput"] input:focus {
        border-color: var(--amber) !important;
        box-shadow: 0 0 0 3px rgba(232,163,23,.25) !important;
    }

    .feet {
        color: var(--muted);
        font-family: "IBM Plex Mono", monospace;
        font-size: .85rem;
        margin: .35rem 0 1.2rem 0;
    }

    .readout {
        background: var(--ink);
        border-radius: 18px;
        padding: 1.7rem 2rem 1.5rem 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(11,31,51,.22);
        margin-bottom: 1.1rem;
    }
    .readout-eyebrow {
        font-family: "IBM Plex Mono", monospace;
        font-size: .74rem;
        letter-spacing: .2em;
        text-transform: uppercase;
        color: #8FA3B8;
        margin-bottom: .3rem;
    }
    .readout-value {
        font-family: "IBM Plex Mono", monospace;
        font-size: 3.6rem;
        font-weight: 600;
        color: #FFFFFF;
        line-height: 1.1;
    }
    .readout-value .unit {
        font-size: 1.4rem;
        font-weight: 500;
        color: var(--amber);
    }
    .readout-margin {
        display: inline-block;
        margin-top: .7rem;
        padding: .25rem .85rem;
        border: 1px solid rgba(232,163,23,.55);
        border-radius: 999px;
        color: var(--amber);
        font-family: "IBM Plex Mono", monospace;
        font-size: .82rem;
    }

    .sentence {
        background: var(--surface);
        border-left: 4px solid var(--amber);
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.25rem;
        color: var(--ink);
        font-size: 1.02rem;
        line-height: 1.55;
        margin-bottom: 1.6rem;
    }

    section[data-testid="stExpander"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 12px;
        margin-bottom: .7rem;
    }
    section[data-testid="stExpander"] summary {
        color: var(--ink) !important;
        font-family: "Space Grotesk", sans-serif;
        font-weight: 600 !important;
        font-size: .98rem !important;
    }
    section[data-testid="stExpander"] p, section[data-testid="stExpander"] li {
        color: var(--ink);
    }
    div[data-testid="stExpander"] p { color: var(--ink); }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }

    .footer {
        margin-top: 1.6rem;
        color: var(--muted);
        font-family: "IBM Plex Mono", monospace;
        font-size: .78rem;
        letter-spacing: .04em;
    }

    @media (max-width: 640px) {
        .hero h1 { font-size: 2rem; }
        .readout-value { font-size: 2.7rem; }
        .block-container { padding-left: 1.1rem; padding-right: 1.1rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        * { transition: none !important; animation: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "best_model.pkl")


@st.cache_data
def load_results():
    return json.loads((ROOT / "model_results.json").read_text(encoding="utf-8"))


model = load_model()
results = load_results()


def predict(height: float) -> float:
    return float(model.predict([[height]])[0])


def feet_inches(height: float) -> str:
    ft = int(height // 12)
    inch = height - ft * 12
    return f"{ft} ft {inch:.1f} in"


st.markdown(
    f"""
    <div class="eyebrow">Height <b>&rarr;</b> Weight</div>
    <div class="hero">
        <h1>Weight Predictor</h1>
        <p>Type a height in inches and instantly see the estimated weight,
        based on measurements of {results['n_rows']:,} people.</p>
    </div>
    <div class="ruler"></div>
    """,
    unsafe_allow_html=True,
)

height = st.number_input(
    "Height in inches",
    min_value=54.0,
    max_value=79.0,
    value=66.0,
    step=0.1,
    format="%.1f",
    help="Enter a value between 54 and 79 inches",
)

rmse = results["best_rmse"]
r2 = results["best_r2"]
prediction = predict(height)

st.markdown(
    f'<div class="feet">That is about {feet_inches(height)}</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="readout">
        <div class="readout-eyebrow">Estimated weight</div>
        <div class="readout-value">{prediction:.0f} <span class="unit">lb</span></div>
        <div class="readout-margin">+/- {rmse:.0f} lb typical margin</div>
    </div>
    <div class="sentence">
        A person <b>{height:.1f} inches</b> tall is usually about
        <b>{prediction:.0f} pounds</b> - give or take <b>{rmse:.0f} pounds</b>.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("How accurate is it?"):
    st.write(
        f"**About {r2 * 100:.0f}% accurate.** The model accounts for roughly "
        f"{r2 * 100:.0f} out of every 100 differences in weight between people, "
        f"and its estimate is typically off by no more than about "
        f"**{rmse:.0f} pounds**."
    )
    st.write(
        "It works best between **54 and 79 inches**, the range it was trained on."
    )

with st.expander("How was this built?"):
    st.write(
        "The same steps as Practical 7B: the data was split into a training part "
        "and a testing part, five regression models were trained, and the "
        "best-scoring one was kept."
    )
    st.write(
        "The winner is a **polynomial (degree 3) linear regression** on height. "
        "No technical steps needed - just type a height above."
    )

with st.expander("Compare all 5 models"):
    rows = [
        {
            "Model": family,
            "Accuracy (R2)": round(r["r2"], 4),
            "Typical error (lbs)": round(r["rmse"], 2),
            "Average miss (lbs)": round(r["mae"], 2),
        }
        for family, r in results["family_best"].items()
    ]
    table = pd.DataFrame(rows).sort_values("Accuracy (R2)", ascending=False)
    st.dataframe(table, hide_index=True, width="stretch")
    st.markdown(
        f"""
        <div class="footer">
        PRACTICAL 7B BASELINES - LINEAR .8577 / LASSO .8577 / RIDGE .8577 /
        TREE .8345 / FOREST .8469 &nbsp;|&nbsp; THIS APP - {results['best_r2']:.4f}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer">WEIGHT-HEIGHT DATASET &middot; 10,000 PEOPLE &middot; '
    'HEIGHTS 54-79 IN &middot; WEIGHTS IN POUNDS</div>',
    unsafe_allow_html=True,
)
