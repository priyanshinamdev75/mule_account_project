import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import plotly.graph_objects as go
import plotly.express as px



st.set_page_config(
    page_title="MuleGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)



MODEL_PATH = "final_mule_account_model.joblib"


FEATURE_COLUMNS = [
    "INIT_BALANCE",
    "IN_COUNT",
    "IN_TOTAL_AMOUNT",
    "IN_MEAN_AMOUNT",
    "IN_MEDIAN_AMOUNT",
    "IN_STD_AMOUNT",
    "IN_MIN_AMOUNT",
    "IN_MAX_AMOUNT",
    "IN_UNIQUE_COUNTERPARTIES",
    "OUT_COUNT",
    "OUT_TOTAL_AMOUNT",
    "OUT_MEAN_AMOUNT",
    "OUT_MEDIAN_AMOUNT",
    "OUT_STD_AMOUNT",
    "OUT_MIN_AMOUNT",
    "OUT_MAX_AMOUNT",
    "OUT_UNIQUE_COUNTERPARTIES",
    "IN_TOTAL_COUNT",
    "OUT_TOTAL_COUNT",
    "TOTAL_TRANSACTION_COUNT",
    "FIRST_SENT_TIME",
    "FIRST_RECEIVED_TIME",
    "LAST_SENT_TIME",
    "LAST_RECEIVED_TIME",
    "FIRST_ACTIVITY_TIME",
    "LAST_ACTIVITY_TIME",
    "ACTIVE_TIME_SPAN",
    "IN_DEGREE",
    "OUT_DEGREE",
    "TOTAL_DEGREE",
    "DEGREE_BALANCE",
    "OUT_RECIPROCAL_COUNT",
    "IN_RECIPROCAL_COUNT",
    "TOTAL_RECIPROCAL_COUNT",
    "OUT_COUNTERPARTY_HHI",
    "IN_COUNTERPARTY_HHI",
    "OUT_MAX_TX_PER_TIME",
    "OUT_MEAN_TX_PER_ACTIVE_TIME",
    "OUT_ACTIVE_TIME_POINTS",
    "IN_MAX_TX_PER_TIME",
    "IN_MEAN_TX_PER_ACTIVE_TIME",
    "IN_ACTIVE_TIME_POINTS",
    "SELF_TRANSFER_COUNT",
    "TOTAL_RECEIVED",
    "TOTAL_SENT",
    "TOTAL_MONEY_FLOW",
    "NET_MONEY_FLOW",
    "IN_OUT_AMOUNT_RATIO",
    "OUT_IN_AMOUNT_RATIO",
    "IN_OUT_COUNT_RATIO",
    "TOTAL_COUNTERPARTIES",
    "RECIPROCAL_TRANSACTION_RATIO",
    "TRANSACTIONS_PER_ACTIVE_TIME",
    "MONEY_FLOW_PER_TRANSACTION",
]



@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    pipeline = joblib.load(MODEL_PATH)


    try:
        imputer = pipeline.named_steps.get("imputer")

        if imputer is not None and not hasattr(imputer, "_fill_dtype"):
            imputer._fill_dtype = np.float64

    except Exception:
        pass

    return pipeline


model = load_model()



def model_predict_proba(X: pd.DataFrame) -> np.ndarray:
    """Return fraud probabilities, with a compatible imputer fallback."""

    try:
        return model.predict_proba(X)

    except AttributeError:

        imputer = model.named_steps["imputer"]
        classifier = model.named_steps["classifier"]

        medians = pd.Series(
            imputer.statistics_,
            index=list(model.feature_names_in_),
        )

        X_filled = X.fillna(medians).astype(float)

        return classifier.predict_proba(X_filled.values)



for key, default in [
    ("analysis_count", 0),
    ("fraud_count", 0),
    ("not_fraud_count", 0),
    ("last_result", None),
    ("history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default



st.markdown(
    """
    <style>

    :root { --ink: #11150f; --muted: #687060; --panel: #ffffff; --line: #e7eadf; --lime: #b7ed1b; --deep: #080b08; }
    .stApp { background: radial-gradient(circle at 92% 7%, rgba(183,237,27,.12), transparent 20rem), #f8faf5; color: var(--ink); }
    .block-container { max-width: 1390px; padding-top: .8rem; padding-bottom: 3.5rem; }
    [data-testid="stHeader"] { background: rgba(248,250,245,.78); backdrop-filter: blur(14px); }
    [data-testid="stSidebar"] { background: #0b0f0a; border-right: 1px solid #293126; }
    [data-testid="stSidebar"] * { color: #f7f9f3; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #f7f9f3 !important; }
    h1, h2, h3 { color: #171717 !important; letter-spacing: -0.035em; }
    h1 { font-size: 2.5rem !important; font-weight: 800 !important; }
    h2 { font-weight: 700 !important; }
    p, .stCaption { color: var(--muted); }
    [data-testid="stMetric"] { background: rgba(255,255,255,.88); border: 1px solid var(--line); border-radius: 18px; padding: 1.05rem; box-shadow: 0 10px 28px rgba(16, 20, 11, .06); transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease; }
    [data-testid="stMetric"]:hover { transform: translateY(-4px); border-color: rgba(183,237,27,.85); box-shadow: 0 16px 36px rgba(53,78,20,.14); }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
    [data-testid="stMetricValue"] { color: #171717 !important; }
    .stButton > button, .stDownloadButton > button { border-radius: 999px; border: 1px solid #1d211c; background: #161b14; color: white; font-weight: 700; min-height: 2.6rem; transition: transform .22s ease, background .22s ease, box-shadow .22s ease; }
    .stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-2px); background: #b7ed1b; color: #12150f; border-color: #b7ed1b; box-shadow: 0 9px 20px rgba(137,190,19,.25); }
    [data-baseweb="tab-list"] { gap: .35rem; padding: .38rem; border: 1px solid #e1e6d6; border-radius: 999px; background: rgba(255,255,255,.78); margin: .8rem 0 1.7rem; overflow-x: auto; }
    [data-baseweb="tab"] { background: transparent; border: 1px solid transparent; border-radius: 999px; color: #65705f; padding: .58rem .9rem; font-weight: 700; white-space: nowrap; transition: color .2s ease, background .2s ease, transform .2s ease; }
    [data-baseweb="tab"]:hover { background: #eef4df; color: #192014; transform: translateY(-1px); }
    [aria-selected="true"][data-baseweb="tab"] { color: #10150d; border-color: #b7ed1b; background: #b7ed1b; box-shadow: 0 4px 12px rgba(101,145,10,.2); }
    [data-testid="stExpander"] { background: #fff; border: 1px solid var(--line); border-radius: 16px; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--line); border-radius: 16px; overflow: hidden; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color: #e6eadc !important; border-radius: 18px !important; }
    .intro { position: relative; overflow: hidden; color: #fff; background-color: var(--deep); background-image: linear-gradient(rgba(183,237,27,.065) 1px, transparent 1px), linear-gradient(90deg, rgba(183,237,27,.065) 1px, transparent 1px), radial-gradient(circle at 77% 28%, rgba(183,237,27,.24), transparent 18rem), radial-gradient(circle at 25% 0%, rgba(67,109,28,.24), transparent 22rem); background-size: 42px 42px, 42px 42px, auto, auto; border: 1px solid #273124; border-radius: 24px; padding: clamp(2rem, 5vw, 4.5rem); margin: .5rem 0 1.1rem; box-shadow: 0 25px 65px rgba(6,10,5,.24); }
    .intro::after { content: ''; position: absolute; width: 240px; height: 240px; border: 1px solid rgba(183,237,27,.34); border-radius: 50%; right: -76px; bottom: -125px; box-shadow: 0 0 0 28px rgba(183,237,27,.04), 0 0 0 57px rgba(183,237,27,.035); }
    .intro-nav { display:flex; align-items:center; justify-content:space-between; gap:1rem; position:relative; z-index:1; margin-bottom: clamp(2.2rem, 6vw, 5.5rem); font-size:.8rem; letter-spacing:.08em; text-transform:uppercase; }
    .brand { display:flex; align-items:center; gap:.55rem; font-weight:800; color:#fff; letter-spacing:0; text-transform:none; font-size:1rem; }.brand-mark { color:var(--lime); font-size:1.35rem; }.intro-live { color:#d5e5c0; border:1px solid rgba(183,237,27,.45); padding:.35rem .7rem; border-radius:999px; }
    .script { color:var(--lime); font-family: 'Brush Script MT', 'Segoe Script', cursive; font-size:clamp(2rem,4.5vw,4rem); font-weight:400; transform:rotate(-4deg); margin-bottom:-.1rem; }
    .intro-title { position:relative; z-index:1; color:#fff; max-width:830px; font-size:clamp(2.7rem,7vw,6.4rem); line-height:.9; letter-spacing:-.075em; font-weight:850; margin:0; }.intro-title em { color:var(--lime); font-style:normal; }
    .intro-copy { position:relative; z-index:1; max-width:600px; color:#d6ddcf; font-size:1.05rem; line-height:1.6; margin:1.35rem 0 1.5rem; }.intro-actions { position:relative; z-index:1; display:flex; align-items:center; gap:.7rem; flex-wrap:wrap; }.intro-action { display:inline-block; padding:.72rem 1rem; color:#10150d; background:var(--lime); border-radius:999px; font-weight:800; font-size:.88rem; }.intro-note { color:#d6ddcf; font-size:.84rem; }
    .hero { background-color: #10150d; background-image: radial-gradient(circle at 85% 20%, rgba(183,237,27,.22), transparent 23rem); border: 1px solid #2c3727; border-radius: 20px; padding: 2.4rem 2.5rem; margin: .5rem 0 1.5rem; box-shadow: 0 18px 42px rgba(12,16,8,.14); }
    .hero-kicker { color: #b7ed1b; font-size: .76rem; letter-spacing: .14em; font-weight: 800; text-transform: uppercase; }
    .hero-title { color: #fff; font-size: clamp(2.3rem, 5vw, 4.3rem); line-height: .98; font-weight: 850; margin: .6rem 0; max-width: 780px; }
    .hero-copy { color: #d1d8c8; max-width: 650px; font-size: 1.03rem; line-height: 1.55; }
    .status-pill { display: inline-block; padding: .32rem .68rem; border-radius: 999px; color: #c9fa43; background: rgba(183,237,27,.08); border: 1px solid rgba(183,237,27,.35); font-size: .78rem; font-weight: 700; }
    @media (max-width: 700px) { .intro { border-radius: 18px; } .intro-nav { margin-bottom: 3rem; } .intro-live { display:none; } [data-baseweb="tab-list"] { border-radius:18px; } }

    </style>
    """,
    unsafe_allow_html=True
)



def derive_features(base: dict) -> dict:

    f = dict(base)

    def div(a, b):
        b = float(b)
        return float(a) / b if b != 0 else 0.0

    in_count = f.get("IN_COUNT", 0)
    out_count = f.get("OUT_COUNT", 0)

    in_total = f.get("IN_TOTAL_AMOUNT", 0)
    out_total = f.get("OUT_TOTAL_AMOUNT", 0)

    in_uniq = f.get("IN_UNIQUE_COUNTERPARTIES", 0)
    out_uniq = f.get("OUT_UNIQUE_COUNTERPARTIES", 0)

    f["IN_MEAN_AMOUNT"] = div(in_total, in_count)
    f["OUT_MEAN_AMOUNT"] = div(out_total, out_count)

    f["IN_TOTAL_COUNT"] = in_count
    f["OUT_TOTAL_COUNT"] = out_count

    total_tx = in_count + out_count
    f["TOTAL_TRANSACTION_COUNT"] = total_tx

    f["IN_DEGREE"] = in_uniq
    f["OUT_DEGREE"] = out_uniq
    f["TOTAL_DEGREE"] = in_uniq + out_uniq
    f["DEGREE_BALANCE"] = in_uniq - out_uniq
    f["TOTAL_COUNTERPARTIES"] = in_uniq + out_uniq

    total_recip = (
        f.get("OUT_RECIPROCAL_COUNT", 0)
        + f.get("IN_RECIPROCAL_COUNT", 0)
    )
    f["TOTAL_RECIPROCAL_COUNT"] = total_recip
    f["RECIPROCAL_TRANSACTION_RATIO"] = div(total_recip, total_tx)

    first_act = f.get("FIRST_ACTIVITY_TIME", 0)
    last_act = f.get("LAST_ACTIVITY_TIME", 0)

    f["ACTIVE_TIME_SPAN"] = last_act - first_act

    f["FIRST_SENT_TIME"] = first_act
    f["FIRST_RECEIVED_TIME"] = first_act
    f["LAST_SENT_TIME"] = last_act
    f["LAST_RECEIVED_TIME"] = last_act

    f["TRANSACTIONS_PER_ACTIVE_TIME"] = div(
        total_tx,
        f["ACTIVE_TIME_SPAN"]
    )

    out_points = f.get("OUT_ACTIVE_TIME_POINTS", 0)
    in_points = f.get("IN_ACTIVE_TIME_POINTS", 0)

    f["OUT_MEAN_TX_PER_ACTIVE_TIME"] = div(out_count, out_points)
    f["IN_MEAN_TX_PER_ACTIVE_TIME"] = div(in_count, in_points)

    f["OUT_MAX_TX_PER_TIME"] = max(
        f.get("OUT_MAX_TX_PER_TIME", 0),
        f["OUT_MEAN_TX_PER_ACTIVE_TIME"]
    )
    f["IN_MAX_TX_PER_TIME"] = max(
        f.get("IN_MAX_TX_PER_TIME", 0),
        f["IN_MEAN_TX_PER_ACTIVE_TIME"]
    )

    f["TOTAL_RECEIVED"] = in_total
    f["TOTAL_SENT"] = out_total
    f["TOTAL_MONEY_FLOW"] = in_total + out_total
    f["NET_MONEY_FLOW"] = in_total - out_total

    f["MONEY_FLOW_PER_TRANSACTION"] = div(
        f["TOTAL_MONEY_FLOW"],
        total_tx
    )

    f["IN_OUT_AMOUNT_RATIO"] = div(in_total, out_total)
    f["OUT_IN_AMOUNT_RATIO"] = div(out_total, in_total)
    f["IN_OUT_COUNT_RATIO"] = div(in_count, out_count)

    return f



def build_feature_row(values: dict) -> pd.DataFrame:
    """Build a model input row from the supplied feature values."""

    row = {col: np.nan for col in FEATURE_COLUMNS}

    for key, value in values.items():
        if key in row and value is not None:
            row[key] = float(value)

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)



def categorize_risk(risk_score: float):

    if risk_score < 20:
        return "LOW RISK", "🟢"

    if risk_score < 50:
        return "MEDIUM RISK", "🟡"

    if risk_score < 75:
        return "HIGH RISK", "🟠"

    return "CRITICAL RISK", "🔴"



def predict_account(values: dict, threshold: float = 0.50) -> dict:
    """Score an account and return its fraud-risk classification."""

    X = build_feature_row(values)

    probability = float(
        model_predict_proba(X)[0][1]
    )

    prediction = "FRAUD" if probability >= threshold else "NOT FRAUD"

    risk_score = probability * 100

    risk_category, risk_icon = categorize_risk(risk_score)

    return {
        "prediction": prediction,
        "probability": probability,
        "risk_score": risk_score,
        "risk_category": risk_category,
        "risk_icon": risk_icon,
        "threshold": threshold,
        "features_provided": int(X.notna().sum(axis=1).iloc[0]),
        "features_imputed": int(X.isna().sum(axis=1).iloc[0]),
        "input_frame": X,
    }



def get_risk_description(category: str) -> str:

    descriptions = {
        "LOW RISK":
            "The account shows relatively normal behavioural characteristics.",

        "MEDIUM RISK":
            "Some behavioural characteristics require additional monitoring.",

        "HIGH RISK":
            "Multiple suspicious behavioural indicators have been detected.",

        "CRITICAL RISK":
            "Strong suspicious characteristics require priority investigation.",
    }

    return descriptions.get(
        category,
        "Review the account carefully."
    )



def create_gauge(probability: float, threshold: float):

    percentage = probability * 100

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=percentage,
            number={
                "suffix": "%",
                "font": {"size": 38}
            },
            title={"text": "Fraud Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#dc2626"},
                "steps": [
                    {"range": [0, 20], "color": "#dcfce7"},
                    {"range": [20, 50], "color": "#fef9c3"},
                    {"range": [50, 75], "color": "#ffedd5"},
                    {"range": [75, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#1f2937", "width": 4},
                    "thickness": 0.85,
                    "value": threshold * 100,
                },
            },
        )
    )

    fig.update_layout(
        height=330,
        margin={"l": 25, "r": 25, "t": 60, "b": 20}
    )

    return fig



@st.cache_data
def get_feature_importance():

    classifier = model.named_steps["classifier"]

    importance = pd.DataFrame({
        "Feature": list(model.feature_names_in_),
        "Importance": classifier.feature_importances_,
    })

    return importance.sort_values(
        "Importance",
        ascending=False
    ).reset_index(drop=True)



PRESET_NORMAL = {
    "INIT_BALANCE": 249.0,
    "IN_COUNT": 80,
    "IN_TOTAL_AMOUNT": 6776575.0,
    "IN_MEDIAN_AMOUNT": 128.0,
    "IN_STD_AMOUNT": 629332.0,
    "IN_MIN_AMOUNT": 12.0,
    "IN_MAX_AMOUNT": 5428735.0,
    "IN_UNIQUE_COUNTERPARTIES": 5,
    "IN_ACTIVE_TIME_POINTS": 70,
    "OUT_COUNT": 128,
    "OUT_TOTAL_AMOUNT": 28287.0,
    "OUT_MEDIAN_AMOUNT": 298.0,
    "OUT_STD_AMOUNT": 0.0,
    "OUT_MIN_AMOUNT": 192.0,
    "OUT_MAX_AMOUNT": 298.0,
    "OUT_UNIQUE_COUNTERPARTIES": 6,
    "OUT_ACTIVE_TIME_POINTS": 128,
    "FIRST_ACTIVITY_TIME": 1,
    "LAST_ACTIVITY_TIME": 199,
    "IN_COUNTERPARTY_HHI": 0.22,
    "OUT_COUNTERPARTY_HHI": 0.22,
    "IN_RECIPROCAL_COUNT": 3,
    "OUT_RECIPROCAL_COUNT": 3,
    "SELF_TRANSFER_COUNT": 0,
}


PRESET_SUSPICIOUS = {
    "INIT_BALANCE": 0.0,
    "IN_COUNT": 96,
    "IN_TOTAL_AMOUNT": 4820000.0,
    "IN_MEDIAN_AMOUNT": 41000.0,
    "IN_STD_AMOUNT": 68000.0,
    "IN_MIN_AMOUNT": 1.0,
    "IN_MAX_AMOUNT": 410000.0,
    "IN_UNIQUE_COUNTERPARTIES": 80,
    "IN_ACTIVE_TIME_POINTS": 6,
    "OUT_COUNT": 11,
    "OUT_TOTAL_AMOUNT": 4815000.0,
    "OUT_MEDIAN_AMOUNT": 430000.0,
    "OUT_STD_AMOUNT": 50000.0,
    "OUT_MIN_AMOUNT": 5.0,
    "OUT_MAX_AMOUNT": 900000.0,
    "OUT_UNIQUE_COUNTERPARTIES": 2,
    "OUT_ACTIVE_TIME_POINTS": 2,
    "FIRST_ACTIVITY_TIME": 142,
    "LAST_ACTIVITY_TIME": 146,
    "IN_COUNTERPARTY_HHI": 0.02,
    "OUT_COUNTERPARTY_HHI": 0.95,
    "IN_RECIPROCAL_COUNT": 0,
    "OUT_RECIPROCAL_COUNT": 0,
    "SELF_TRANSFER_COUNT": 0,
}


if "form_values" not in st.session_state:
    st.session_state.form_values = dict(PRESET_NORMAL)



with st.sidebar:

    st.title("🛡️ MuleGuard AI")

    st.write("Fraud & Suspicious Mule Account Detection")

    st.divider()

    st.subheader("System Status")

    if model is None:
        st.error("Model NOT loaded")
        st.caption(
            f"`{MODEL_PATH}` app.py ke saath same folder me rakho."
        )
    else:
        st.success("Model Loaded")
        st.caption(
            f"{len(FEATURE_COLUMNS)} features · RandomForest pipeline"
        )

    st.divider()

    st.subheader("Decision Threshold")

    threshold = st.slider(
        "Fraud threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Accounts at or above this probability are classified as fraud.",
    )

    st.divider()

    st.subheader("Session Stats")

    st.write(f"Analyses: {st.session_state.analysis_count}")
    st.write(f"Flagged FRAUD: {st.session_state.fraud_count}")
    st.write(f"NOT FRAUD: {st.session_state.not_fraud_count}")

    if st.button("Reset session stats", use_container_width=True):
        st.session_state.analysis_count = 0
        st.session_state.fraud_count = 0
        st.session_state.not_fraud_count = 0
        st.session_state.history = []
        st.rerun()



st.markdown(
    """
    <section class="intro">
      <div class="intro-nav">
        <div class="brand"><span class="brand-mark">◈</span> MuleGuard <span style="color:#aab6a2;font-weight:500;">AI</span></div>
        <div class="intro-live">● Protected workspace</div>
      </div>
      <div class="script">Intelligence, with purpose.</div>
      <div class="intro-title">SEE THE SIGNAL.<br><em>STOP</em> THE FLOW.</div>
      <div class="intro-copy">A calm, focused command centre for revealing suspicious money movement before it becomes financial harm. Turn account behaviour into confident investigation decisions.</div>
      <div class="intro-actions"><span class="intro-action">Explore risk signals ↓</span><span class="intro-note">54 behavioural features · Random Forest intelligence</span></div>
    </section>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">Your investigation workspace</div>
      <div class="hero-title">MOVE FROM DATA<br>TO <span style="color:#b7ed1b;">DECISIONS.</span></div>
      <div class="hero-copy">Review account behaviour, identify elevated mule-account risk, and prioritize the next action with model-led intelligence.</div>
      <div style="margin-top:1rem;"><span class="status-pill">● Secure model connection active</span></div>
    </section>
    """,
    unsafe_allow_html=True,
)



if model is None:

    st.error(
        f"""
        **Model file nahi mili: `{MODEL_PATH}`**

        `final_mule_account_model.joblib` ko `app.py` ke saath
        same folder me rakho aur app dobara chalao.
        """
    )

    st.stop()



st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model Status", "Ready")

with col2:
    st.metric("Analyses", st.session_state.analysis_count)

with col3:
    st.metric("Elevated Risk", st.session_state.fraud_count)

with col4:
    st.metric("Clear", st.session_state.not_fraud_count)



(
    dashboard_tab,
    analysis_tab,
    batch_tab,
    performance_tab,
    intelligence_tab,
    about_tab,
) = st.tabs(
    [
        "🏠 Dashboard",
        "🔍 Account Analysis",
        "📁 Batch Scoring",
        "📊 Model Performance",
        "🛡️ Risk Intelligence",
        "ℹ️ About Project",
    ]
)



with dashboard_tab:

    st.header("Mule Account Detection Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🔍 Account Detection")
        st.write(
            """
            Analyze account behaviour and classify the account as a
            potential mule account or a normal account.
            """
        )

    with col2:
        st.subheader("📊 Risk Scoring")
        st.write(
            """
            The model outputs a fraud probability which is converted
            into a 0–100 risk score and a risk category.
            """
        )

    with col3:
        st.subheader("🛡️ Investigation Support")
        st.write(
            """
            High-risk accounts are flagged with the specific behavioural
            indicators that drove the decision.
            """
        )

    st.divider()

    st.header("How the System Works")

    process_col1, process_col2 = st.columns(2)

    with process_col1:
        st.markdown(
            """
            **1. Account Feature Input**

            Account-level behavioural aggregates are entered or uploaded.

            **2. Feature Derivation**

            Ratios, degrees and velocity features are computed automatically.

            **3. Median Imputation**

            Missing features are filled using the training-set medians.
            """
        )

    with process_col2:
        st.markdown(
            """
            **4. Random Forest Prediction**

            The trained pipeline outputs a fraud probability.

            **5. Threshold & Risk Score**

            Probability is compared against the decision threshold.

            **6. Investigation**

            Flagged accounts are queued for AML review.
            """
        )

    st.divider()

    if st.session_state.history:

        st.header("Recent Analyses (this session)")

        st.dataframe(
            pd.DataFrame(st.session_state.history),
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "Account Analysis"
        )



with analysis_tab:

    st.header("🔍 Account Analysis")

    st.write(
        """
        Enter the account's behavioural aggregates below. Derived features
        (ratios, degrees, money flow, velocity) are computed automatically.
        """
    )


    preset_col1, preset_col2, preset_col3 = st.columns([1, 1, 2])

    with preset_col1:
        if st.button("Load normal account", use_container_width=True):
            st.session_state.form_values = dict(PRESET_NORMAL)
            st.rerun()

    with preset_col2:
        if st.button("Load suspicious account", use_container_width=True):
            st.session_state.form_values = dict(PRESET_SUSPICIOUS)
            st.rerun()

    with preset_col3:
        st.caption("Select a preset to populate the form.")

    fv = st.session_state.form_values

    st.divider()


    st.subheader("Account Basics")

    col1, col2, col3 = st.columns(3)

    with col1:
        init_balance = st.number_input(
            "Initial Balance",
            min_value=0.0,
            value=float(fv["INIT_BALANCE"]),
            step=500.0,
        )

    with col2:
        first_activity = st.number_input(
            "First Activity Time",
            min_value=0,
            value=int(fv["FIRST_ACTIVITY_TIME"]),
            step=1,
            help="Time index of the account's first transaction.",
        )

    with col3:
        last_activity = st.number_input(
            "Last Activity Time",
            min_value=0,
            value=int(fv["LAST_ACTIVITY_TIME"]),
            step=1,
            help="Time index of the account's last transaction.",
        )


    st.subheader("Incoming Transactions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        in_count = st.number_input(
            "Incoming Count",
            min_value=0,
            value=int(fv["IN_COUNT"]),
            step=1,
        )

    with col2:
        in_total = st.number_input(
            "Incoming Total Amount",
            min_value=0.0,
            value=float(fv["IN_TOTAL_AMOUNT"]),
            step=1000.0,
        )

    with col3:
        in_median = st.number_input(
            "Incoming Median Amount",
            min_value=0.0,
            value=float(fv["IN_MEDIAN_AMOUNT"]),
            step=100.0,
        )

    with col4:
        in_std = st.number_input(
            "Incoming Std Amount",
            min_value=0.0,
            value=float(fv["IN_STD_AMOUNT"]),
            step=100.0,
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        in_min = st.number_input(
            "Incoming Min Amount",
            min_value=0.0,
            value=float(fv["IN_MIN_AMOUNT"]),
            step=100.0,
        )

    with col2:
        in_max = st.number_input(
            "Incoming Max Amount",
            min_value=0.0,
            value=float(fv["IN_MAX_AMOUNT"]),
            step=1000.0,
        )

    with col3:
        in_uniq = st.number_input(
            "Incoming Unique Counterparties",
            min_value=0,
            value=int(fv["IN_UNIQUE_COUNTERPARTIES"]),
            step=1,
        )

    with col4:
        in_points = st.number_input(
            "Incoming Active Time Points",
            min_value=0,
            value=int(fv["IN_ACTIVE_TIME_POINTS"]),
            step=1,
            help="Kitne distinct time-steps pe paisa aaya.",
        )


    st.subheader("Outgoing Transactions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        out_count = st.number_input(
            "Outgoing Count",
            min_value=0,
            value=int(fv["OUT_COUNT"]),
            step=1,
        )

    with col2:
        out_total = st.number_input(
            "Outgoing Total Amount",
            min_value=0.0,
            value=float(fv["OUT_TOTAL_AMOUNT"]),
            step=1000.0,
        )

    with col3:
        out_median = st.number_input(
            "Outgoing Median Amount",
            min_value=0.0,
            value=float(fv["OUT_MEDIAN_AMOUNT"]),
            step=100.0,
        )

    with col4:
        out_std = st.number_input(
            "Outgoing Std Amount",
            min_value=0.0,
            value=float(fv["OUT_STD_AMOUNT"]),
            step=100.0,
            help="Model ke top-2 features me se ek.",
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        out_min = st.number_input(
            "Outgoing Min Amount",
            min_value=0.0,
            value=float(fv["OUT_MIN_AMOUNT"]),
            step=100.0,
            help="Model ka sabse important feature.",
        )

    with col2:
        out_max = st.number_input(
            "Outgoing Max Amount",
            min_value=0.0,
            value=float(fv["OUT_MAX_AMOUNT"]),
            step=1000.0,
        )

    with col3:
        out_uniq = st.number_input(
            "Outgoing Unique Counterparties",
            min_value=0,
            value=int(fv["OUT_UNIQUE_COUNTERPARTIES"]),
            step=1,
        )

    with col4:
        out_points = st.number_input(
            "Outgoing Active Time Points",
            min_value=0,
            value=int(fv["OUT_ACTIVE_TIME_POINTS"]),
            step=1,
        )


    st.subheader("Network Behaviour")

    col1, col2, col3 = st.columns(3)

    with col1:
        in_hhi = st.number_input(
            "Incoming Counterparty HHI",
            min_value=0.0,
            max_value=1.0,
            value=float(fv["IN_COUNTERPARTY_HHI"]),
            step=0.01,
            help="0 = paisa bahut saare logon se aaya, 1 = sirf ek se.",
        )

    with col2:
        out_hhi = st.number_input(
            "Outgoing Counterparty HHI",
            min_value=0.0,
            max_value=1.0,
            value=float(fv["OUT_COUNTERPARTY_HHI"]),
            step=0.01,
            help="1 ke paas = saara paisa ek hi account ko gaya (mule signal).",
        )

    with col3:
        self_transfer = st.number_input(
            "Self Transfer Count",
            min_value=0,
            value=int(fv["SELF_TRANSFER_COUNT"]),
            step=1,
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        in_recip = st.number_input(
            "Incoming Reciprocal Count",
            min_value=0,
            value=int(fv["IN_RECIPROCAL_COUNT"]),
            step=1,
        )

    with col2:
        out_recip = st.number_input(
            "Outgoing Reciprocal Count",
            min_value=0,
            value=int(fv["OUT_RECIPROCAL_COUNT"]),
            step=1,
        )

    st.divider()


    analyze_button = st.button(
        "🔍 ANALYZE ACCOUNT",
        type="primary",
        use_container_width=True,
    )

    if analyze_button:

        base_values = {
            "INIT_BALANCE": init_balance,
            "FIRST_ACTIVITY_TIME": first_activity,
            "LAST_ACTIVITY_TIME": last_activity,

            "IN_COUNT": in_count,
            "IN_TOTAL_AMOUNT": in_total,
            "IN_MEDIAN_AMOUNT": in_median,
            "IN_STD_AMOUNT": in_std,
            "IN_MIN_AMOUNT": in_min,
            "IN_MAX_AMOUNT": in_max,
            "IN_UNIQUE_COUNTERPARTIES": in_uniq,
            "IN_ACTIVE_TIME_POINTS": in_points,

            "OUT_COUNT": out_count,
            "OUT_TOTAL_AMOUNT": out_total,
            "OUT_MEDIAN_AMOUNT": out_median,
            "OUT_STD_AMOUNT": out_std,
            "OUT_MIN_AMOUNT": out_min,
            "OUT_MAX_AMOUNT": out_max,
            "OUT_UNIQUE_COUNTERPARTIES": out_uniq,
            "OUT_ACTIVE_TIME_POINTS": out_points,

            "IN_COUNTERPARTY_HHI": in_hhi,
            "OUT_COUNTERPARTY_HHI": out_hhi,
            "IN_RECIPROCAL_COUNT": in_recip,
            "OUT_RECIPROCAL_COUNT": out_recip,
            "SELF_TRANSFER_COUNT": self_transfer,
        }

        full_values = derive_features(base_values)

        result = predict_account(full_values, threshold=threshold)

        prediction = result["prediction"]
        probability = result["probability"]
        risk_score = result["risk_score"]
        risk_category = result["risk_category"]


        st.session_state.analysis_count += 1

        if prediction == "FRAUD":
            st.session_state.fraud_count += 1
        else:
            st.session_state.not_fraud_count += 1

        st.session_state.last_result = result

        st.session_state.history.append({
            "#": st.session_state.analysis_count,
            "Prediction": prediction,
            "Probability": f"{probability * 100:.2f}%",
            "Risk Category": risk_category,
            "Threshold": f"{threshold:.2f}",
        })


        st.divider()

        st.header("🚨 Detection Result")

        result_col1, result_col2 = st.columns([1.2, 1])

        with result_col1:
            st.plotly_chart(
                create_gauge(probability, threshold),
                use_container_width=True,
            )

        with result_col2:
            st.metric("Prediction", prediction)
            st.metric("Fraud Probability", f"{probability * 100:.2f}%")
            st.metric("Risk Score", f"{risk_score:.2f}/100")
            st.metric("Risk Category", risk_category)

        if prediction == "FRAUD":
            st.error(
                "🚨 Potential MULE ACCOUNT detected. "
                "Further investigation is recommended."
            )
        else:
            st.success("✅ Account classified as NOT FRAUD.")

        st.info(get_risk_description(risk_category))

        st.caption(
            f"Features provided: {result['features_provided']}/54 · "
            f"Median-imputed: {result['features_imputed']}/54 · "
            f"Decision threshold: {threshold:.2f}"
        )


        st.subheader("Observed Behavioural Indicators")

        indicators = []

        if out_hhi >= 0.70:
            indicators.append(
                f"Outgoing funds are highly concentrated "
                f"(HHI {out_hhi:.2f}) — most money went to very few accounts."
            )

        if in_uniq >= 20 and out_uniq <= 5:
            indicators.append(
                f"Fan-in / fan-out pattern: money received from {in_uniq} "
                f"sources but sent to only {out_uniq}."
            )

        span = last_activity - first_activity

        if span > 0 and (in_count + out_count) / span > 5:
            indicators.append(
                f"High transaction velocity: {in_count + out_count} "
                f"transactions across a span of {span} time units."
            )

        if in_total > 0:
            pass_through = out_total / in_total
            if 0.90 <= pass_through <= 1.10:
                indicators.append(
                    f"Pass-through behaviour: {pass_through * 100:.1f}% of "
                    f"received funds were forwarded onward."
                )

        if init_balance < 1000 and in_total > 500000:
            indicators.append(
                "Very low opening balance combined with large volume "
                "moving through the account."
            )

        if in_recip + out_recip == 0 and (in_uniq + out_uniq) > 10:
            indicators.append(
                "No reciprocal relationships — counterparties never "
                "transact back, which is unusual for genuine activity."
            )

        if indicators:
            for indicator in indicators:
                st.write("• " + indicator)
        else:
            st.write(
                "• No strong rule-based behavioural indicators were triggered."
            )

        st.caption(
            "These indicators provide additional context for the model result."
        )


        with st.expander("View all 54 features sent to the model"):

            display_frame = result["input_frame"].T.reset_index()
            display_frame.columns = ["Feature", "Value"]

            display_frame["Source"] = np.where(
                display_frame["Feature"].isin(base_values.keys()),
                "User input",
                np.where(
                    display_frame["Value"].isna(),
                    "Median-imputed",
                    "Derived",
                ),
            )

            st.dataframe(
                display_frame,
                use_container_width=True,
                hide_index=True,
                height=420,
            )


        st.subheader("Recommended Action")

        if prediction == "FRAUD":
            st.warning(
                """
                Priority review recommended.

                Suggested investigation:

                • Review the full account transaction history

                • Check transaction velocity and timing patterns

                • Verify counterparty relationships

                • Review any previous suspicious activity reports

                • Perform appropriate AML/KYC checks
                """
            )
        else:
            st.success(
                """
                No suspicious signal was generated by the model.
                Continue normal monitoring.
                """
            )


        report = result["input_frame"].copy()
        report["FRAUD_PROBABILITY"] = probability
        report["RISK_SCORE"] = risk_score
        report["RISK_CATEGORY"] = risk_category
        report["THRESHOLD"] = threshold
        report["PREDICTION"] = prediction

        st.download_button(
            "⬇️ Download Detection Report",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="mule_detection_report.csv",
            mime="text/csv",
            use_container_width=True,
        )



with batch_tab:

    st.header("📁 Batch Scoring")

    st.write(
        """
        Upload a CSV of account-level features (the same feature set used
        during training). Every row is scored by the model.
        """
    )

    st.info(
         
        "accurate prediction "
    )


    template = pd.DataFrame(columns=FEATURE_COLUMNS)

    st.download_button(
        "⬇️ Download blank feature template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="muleguard_feature_template.csv",
        mime="text/csv",
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload account features CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        try:
            raw_df = pd.read_csv(uploaded_file)

        except Exception as error:
            st.error(f"CSV read nahi hui: {error}")
            raw_df = None

        if raw_df is not None:

            st.success(
                f"Loaded {len(raw_df)} rows × {len(raw_df.columns)} columns."
            )

            matched = [
                c for c in FEATURE_COLUMNS
                if c in raw_df.columns
            ]

            missing = [
                c for c in FEATURE_COLUMNS
                if c not in raw_df.columns
            ]

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Matched features", f"{len(matched)}/54")

            with col2:
                st.metric("Will be imputed", f"{len(missing)}/54")

            if missing:
                with st.expander("Missing columns"):
                    st.write(", ".join(missing))

            if not matched:
                st.error(
                    "Ek bhi expected feature column match nahi hua. "
                    "Template download karke column names check karo."
                )

            else:

                if st.button(
                    "▶️ Score all rows",
                    type="primary",
                    use_container_width=True,
                ):

                    X = raw_df.reindex(columns=FEATURE_COLUMNS)

                    probabilities = model_predict_proba(X)[:, 1]

                    scored = raw_df.copy()

                    scored["FRAUD_PROBABILITY"] = probabilities
                    scored["RISK_SCORE"] = probabilities * 100

                    scored["PREDICTION"] = np.where(
                        probabilities >= threshold,
                        "FRAUD",
                        "NOT FRAUD",
                    )

                    scored["RISK_CATEGORY"] = pd.cut(
                        scored["RISK_SCORE"],
                        bins=[-0.01, 20, 50, 75, 100.01],
                        labels=[
                            "LOW RISK",
                            "MEDIUM RISK",
                            "HIGH RISK",
                            "CRITICAL RISK",
                        ],
                    )

                    flagged = int((scored["PREDICTION"] == "FRAUD").sum())

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Accounts scored", len(scored))

                    with col2:
                        st.metric("Flagged as FRAUD", flagged)

                    with col3:
                        st.metric(
                            "Flag rate",
                            f"{flagged / len(scored) * 100:.1f}%",
                        )

                    st.subheader("Risk Distribution")

                    distribution = (
                        scored["RISK_CATEGORY"]
                        .value_counts()
                        .reindex([
                            "LOW RISK",
                            "MEDIUM RISK",
                            "HIGH RISK",
                            "CRITICAL RISK",
                        ])
                        .fillna(0)
                        .reset_index()
                    )
                    distribution.columns = ["Risk Category", "Accounts"]

                    st.plotly_chart(
                        px.bar(
                            distribution,
                            x="Risk Category",
                            y="Accounts",
                            color="Risk Category",
                            color_discrete_map={
                                "LOW RISK": "#16a34a",
                                "MEDIUM RISK": "#eab308",
                                "HIGH RISK": "#f97316",
                                "CRITICAL RISK": "#dc2626",
                            },
                        ),
                        use_container_width=True,
                    )

                    st.subheader("Highest Risk Accounts")

                    st.dataframe(
                        scored.sort_values(
                            "FRAUD_PROBABILITY",
                            ascending=False,
                        ).head(25),
                        use_container_width=True,
                    )

                    st.download_button(
                        "⬇️ Download scored results",
                        data=scored.to_csv(index=False).encode("utf-8"),
                        file_name="muleguard_scored_accounts.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )



with performance_tab:

    st.header("📊 Model Performance")


    st.subheader("Model Configuration")

    classifier = model.named_steps["classifier"]

    config = pd.DataFrame({
        "Property": [
            "Algorithm",
            "Number of Trees",
            "Max Depth",
            "Max Features",
            "Min Samples Split",
            "Class Weight",
            "Random State",
            "Preprocessing",
            "Input Features",
        ],
        "Value": [
            type(classifier).__name__,
            str(classifier.n_estimators),
            str(classifier.max_depth),
            str(classifier.max_features),
            str(classifier.min_samples_split),
            str(classifier.class_weight),
            str(classifier.random_state),
            "SimpleImputer(strategy='median')",
            str(len(FEATURE_COLUMNS)),
        ],
    })

    st.dataframe(config, use_container_width=True, hide_index=True)

    st.divider()


    st.subheader("Feature Importance")

    importance = get_feature_importance()

    top_n = st.slider(
        "Number of features to display",
        min_value=5,
        max_value=54,
        value=20,
        step=5,
    )

    top_features = importance.head(top_n).sort_values("Importance")

    st.plotly_chart(
        px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            height=max(400, top_n * 22),
            color="Importance",
            color_continuous_scale="Reds",
        ),
        use_container_width=True,
    )

    with st.expander("Full feature importance table"):
        st.dataframe(
            importance,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()


    st.subheader("Evaluate on a Labelled Test Set")

    st.write(
        """
        Upload a labelled CSV to compute genuine metrics. The file must
        contain the feature columns plus one label column (0 = normal,
        1 = mule).
        """
    )

    test_file = st.file_uploader(
        "Upload labelled test CSV",
        type=["csv"],
        key="test_upload",
    )

    if test_file is not None:

        try:
            test_df = pd.read_csv(test_file)

        except Exception as error:
            st.error(f"CSV read nahi hui: {error}")
            test_df = None

        if test_df is not None:

            label_column = st.selectbox(
                "Select the label column",
                list(test_df.columns),
            )

            if st.button("▶️ Compute metrics", type="primary"):

                from sklearn.metrics import (
                    accuracy_score,
                    precision_score,
                    recall_score,
                    f1_score,
                    roc_auc_score,
                    average_precision_score,
                    confusion_matrix,
                )

                y_true = test_df[label_column].astype(int)

                X_test = test_df.reindex(columns=FEATURE_COLUMNS)

                y_prob = model_predict_proba(X_test)[:, 1]
                y_pred = (y_prob >= threshold).astype(int)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Accuracy",
                        f"{accuracy_score(y_true, y_pred) * 100:.2f}%",
                    )

                with col2:
                    st.metric(
                        "Precision",
                        f"{precision_score(y_true, y_pred, zero_division=0) * 100:.2f}%",
                    )

                with col3:
                    st.metric(
                        "Recall",
                        f"{recall_score(y_true, y_pred, zero_division=0) * 100:.2f}%",
                    )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "F1 Score",
                        f"{f1_score(y_true, y_pred, zero_division=0) * 100:.2f}%",
                    )

                with col2:
                    try:
                        st.metric(
                            "ROC-AUC",
                            f"{roc_auc_score(y_true, y_prob) * 100:.2f}%",
                        )
                    except ValueError:
                        st.metric("ROC-AUC", "N/A")

                with col3:
                    try:
                        st.metric(
                            "PR-AUC",
                            f"{average_precision_score(y_true, y_prob) * 100:.2f}%",
                        )
                    except ValueError:
                        st.metric("PR-AUC", "N/A")

                st.subheader("Confusion Matrix")

                matrix = confusion_matrix(y_true, y_pred)

                st.plotly_chart(
                    px.imshow(
                        matrix,
                        text_auto=True,
                        labels={
                            "x": "Predicted",
                            "y": "Actual",
                            "color": "Count",
                        },
                        x=["NOT FRAUD", "FRAUD"],
                        y=["NOT FRAUD", "FRAUD"],
                        color_continuous_scale="Reds",
                    ),
                    use_container_width=True,
                )

                st.caption(
                    f"Metrics computed at threshold {threshold:.2f}. "
                    # "Sidebar se threshold badalke dobara compute karo."
                )

    else:
        st.info("Upload a labelled test set to calculate performance metrics.")



with intelligence_tab:

    st.header("🛡️ Risk Intelligence")

    st.write(
        "The following risk categories are derived from the model's "
        "predicted fraud probability."
    )

    risk_table = pd.DataFrame({
        "Risk Category": [
            "LOW RISK",
            "MEDIUM RISK",
            "HIGH RISK",
            "CRITICAL RISK",
        ],
        "Risk Score": [
            "0 – 20",
            "20 – 50",
            "50 – 75",
            "75 – 100",
        ],
        "Recommended Action": [
            "Normal Monitoring",
            "Monitor",
            "Investigate",
            "Priority Investigation",
        ],
    })

    st.dataframe(risk_table, use_container_width=True, hide_index=True)

    st.divider()

    st.header("Top Model-Driven Risk Signals")

    st.write(
        "The features below carry the most weight in the trained model:"
    )

    for _, row in get_feature_importance().head(10).iterrows():
        st.write(
            f"• **{row['Feature']}** — importance {row['Importance']:.4f}"
        )

    st.divider()

    st.header("Known Mule Account Indicators")

    indicators = [
        "High transaction velocity over a short active window",
        "Large amounts passing through with near-zero net retention",
        "Fan-in from many sources, fan-out to very few destinations",
        "Highly concentrated outgoing counterparties (high HHI)",
        "Very low initial balance with high total money flow",
        "Absence of reciprocal counterparty relationships",
        "Rapid forwarding of received funds",
        "Uniform outgoing amounts (low outgoing standard deviation)",
        "Sudden activation after a long dormant period",
    ]

    for indicator in indicators:
        st.write("• " + indicator)

    st.info(
        """
        These indicators represent risk signals only.
        A risk prediction should not automatically be interpreted
        as proof of criminal activity.
        """
    )



with about_tab:

    st.header("ℹ️ About MuleGuard AI")

    st.subheader("Problem Statement")

    st.write("AI/ML-Based Classification of Suspicious Mule Accounts")

    st.write(
        """
        Financial institutions process millions of transactions. Some accounts
        are used to receive, hold, and forward illicit funds on behalf of
        others — these are known as mule accounts.

        The objective of this project is to use Machine Learning and
        behavioural network analysis to identify potentially suspicious
        accounts from their transaction patterns.
        """
    )

    st.subheader("Project Pipeline")

    pipeline = pd.DataFrame({
        "Stage": list(range(1, 9)),
        "Process": [
            "Data Collection",
            "Data Exploration",
            "Data Preprocessing",
            "Feature Engineering (54 account-level features)",
            "Model Training (Random Forest, class_weight='balanced')",
            "Model Evaluation",
            "Threshold & Risk Scoring",
            "Streamlit Deployment",
        ],
    })

    st.dataframe(pipeline, use_container_width=True, hide_index=True)

    st.subheader("Deployed Model")

    st.write(
        f"""
        • Pipeline: `SimpleImputer(median)` → `RandomForestClassifier`

        • Trees: {classifier.n_estimators} · Max depth: {classifier.max_depth}

        • Class weight: `balanced` (mule accounts are a minority class)

        • Input: {len(FEATURE_COLUMNS)} account-level behavioural features

        • Output: probability of the account being a mule account
        """
    )

    st.subheader("Future Scope")

    future_scope = [
        "Real-time transaction monitoring",
        "Graph Neural Networks for network-level mule ring detection",
        "Anomaly detection and autoencoders",
        "SHAP-based per-account explainability",
        "Real-time banking system integration",
        "AML rule engine integration",
        "Concept drift monitoring",
        "Investigator feedback loop for continuous retraining",
    ]

    for item in future_scope:
        st.write("• " + item)

    st.divider()

    st.warning(
        """
        Ensure that the feature calculations match those used when training
        the model.
        """
    )



st.divider()

st.caption(
    "🛡️ MuleGuard AI | AI/ML-Based Fraud & Suspicious Mule Account Detection"
)

st.caption(
    "Predictions are produced by the trained Random Forest model. "
    "Identical inputs always produce identical outputs."
)
