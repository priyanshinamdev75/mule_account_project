# =============================================================================
# MULEGUARD AI
# AI/ML-BASED FRAUD & SUSPICIOUS MULE ACCOUNT DETECTION
# STREAMLIT DEMO APPLICATION
# =============================================================================

import streamlit as st
import random
import pandas as pd
import numpy as np
import time


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="MuleGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# DEMO CONFIGURATION
# =============================================================================

DEMO_MODE = True

# Approximately 70% NOT FRAUD
NOT_FRAUD_RATE = 0.70

# Approximately 30% FRAUD
FRAUD_RATE = 0.30


# =============================================================================
# FIXED DEMO MODEL PERFORMANCE
# =============================================================================
#
# These are demonstration values only.
# They are NOT actual ML model performance.
#
# They remain fixed so the dashboard does not show different numbers
# every time Streamlit reruns.
# =============================================================================

DEMO_METRICS = {
    "Logistic Regression": {
        "Accuracy": 91.84,
        "Precision": 87.63,
        "Recall": 84.91,
        "F1 Score": 86.25,
        "ROC-AUC": 94.72,
        "PR-AUC": 82.46
    },

    "Random Forest": {
        "Accuracy": 94.67,
        "Precision": 91.28,
        "Recall": 88.73,
        "F1 Score": 89.99,
        "ROC-AUC": 97.18,
        "PR-AUC": 88.42
    },

    "XGBoost": {
        "Accuracy": 96.31,
        "Precision": 94.16,
        "Recall": 92.47,
        "F1 Score": 93.30,
        "ROC-AUC": 98.42,
        "PR-AUC": 92.18
    }
}


# =============================================================================
# SESSION STATE
# =============================================================================

if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0


if "fraud_count" not in st.session_state:
    st.session_state.fraud_count = 0


if "not_fraud_count" not in st.session_state:
    st.session_state.not_fraud_count = 0


if "last_result" not in st.session_state:
    st.session_state.last_result = None


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 42px !important;
        font-weight: 800 !important;
    }

    h2 {
        font-weight: 750 !important;
    }

    h3 {
        font-weight: 700 !important;
    }

    .small-note {
        color: #667085;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# FUNCTION: GENERATE TRANSACTION ID
# =============================================================================

def generate_transaction_id():

    return (
        "TXN-"
        + str(
            random.randint(
                10000000,
                99999999
            )
        )
    )


# =============================================================================
# FUNCTION: GENERATE DEMO PREDICTION
# =============================================================================

def generate_demo_prediction():

    random_number = random.random()

    # -------------------------------------------------------------------------
    # NOT FRAUD
    # -------------------------------------------------------------------------

    if random_number < NOT_FRAUD_RATE:

        prediction = "NOT FRAUD"

        probability = random.uniform(
            0.04,
            0.29
        )

        risk_category = "LOW RISK"

        risk_icon = "🟢"

    # -------------------------------------------------------------------------
    # FRAUD
    # -------------------------------------------------------------------------

    else:

        prediction = "FRAUD"

        probability = random.uniform(
            0.71,
            0.98
        )

        risk_category = random.choice(
            [
                "HIGH RISK",
                "CRITICAL RISK"
            ]
        )

        risk_icon = "🔴"

    risk_score = probability * 100

    return {
        "prediction": prediction,
        "probability": probability,
        "risk_score": risk_score,
        "risk_category": risk_category,
        "risk_icon": risk_icon
    }


# =============================================================================
# FUNCTION: RISK DESCRIPTION
# =============================================================================

def get_risk_description(category):

    descriptions = {

        "LOW RISK":
            "The transaction shows relatively normal behavioural characteristics.",

        "MEDIUM RISK":
            "Some behavioural characteristics require additional monitoring.",

        "HIGH RISK":
            "Multiple suspicious behavioural indicators have been detected.",

        "CRITICAL RISK":
            "Strong suspicious characteristics require priority investigation."
    }

    return descriptions.get(
        category,
        "Review the transaction carefully."
    )


# =============================================================================
# FUNCTION: GAUGE
# =============================================================================

def create_gauge(probability):

    import plotly.graph_objects as go

    percentage = probability * 100

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=percentage,
            number={
                "suffix": "%",
                "font": {
                    "size": 38
                }
            },
            title={
                "text": "Fraud Probability"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "bar": {
                    "color": "#dc2626"
                },
                "steps": [
                    {
                        "range": [0, 20],
                        "color": "#dcfce7"
                    },
                    {
                        "range": [20, 50],
                        "color": "#fef9c3"
                    },
                    {
                        "range": [50, 75],
                        "color": "#ffedd5"
                    },
                    {
                        "range": [75, 100],
                        "color": "#fee2e2"
                    }
                ]
            }
        )
    )

    fig.update_layout(
        height=330,
        margin={
            "l": 25,
            "r": 25,
            "t": 60,
            "b": 20
        }
    )

    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.title("🛡️ MuleGuard AI")

    st.write(
        "Fraud & Suspicious Mule Account Detection"
    )

    st.divider()

    st.subheader("Navigation")

    st.write("🏠 Dashboard")
    st.write("🔍 Transaction Analysis")
    st.write("📊 Model Performance")
    st.write("🛡️ Risk Intelligence")
    st.write("ℹ️ About Project")

    st.divider()

    st.subheader("System Status")

    st.success("System Online")

    st.warning("Demo Mode Active")

    st.divider()

    st.subheader("Demo Prediction Distribution")

    st.write("NOT FRAUD: approximately 70%")

    st.write("FRAUD: approximately 30%")

    st.caption(
        "Predictions are randomly generated for demonstration."
    )


# =============================================================================
# HEADER
# =============================================================================

st.title(
    "🛡️ MuleGuard AI"
)

st.subheader(
    "AI/ML-Based Classification of Suspicious Mule Accounts"
)

st.write(
    """
    An interactive financial transaction risk analysis platform
    designed to identify potentially suspicious transactions and
    mule-account behaviour.
    """
)

st.info(
    "DEMO MODE: Predictions shown by this application are randomly generated "
    "and are not produced by the trained ML model."
)


# =============================================================================
# TOP DASHBOARD METRICS
# =============================================================================

st.divider()

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "System Status",
        "ONLINE"
    )


with col2:

    st.metric(
        "Analyses Performed",
        st.session_state.analysis_count
    )


with col3:

    st.metric(
        "Fraud Detected",
        st.session_state.fraud_count
    )


with col4:

    st.metric(
        "Not Fraud",
        st.session_state.not_fraud_count
    )


# =============================================================================
# TABS
# =============================================================================

dashboard_tab, analysis_tab, performance_tab, intelligence_tab, about_tab = st.tabs(
    [
        "🏠 Dashboard",
        "🔍 Transaction Analysis",
        "📊 Model Performance",
        "🛡️ Risk Intelligence",
        "ℹ️ About Project"
    ]
)


# =============================================================================
# DASHBOARD TAB
# =============================================================================

with dashboard_tab:

    st.header(
        "Financial Fraud Detection Dashboard"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("🔍 Transaction Detection")

        st.write(
            """
            Analyze financial transactions and classify them
            as potentially fraudulent or not fraudulent.
            """
        )

    with col2:

        st.subheader("📊 Risk Scoring")

        st.write(
            """
            Generate a fraud probability and convert it into
            an easy-to-understand risk score.
            """
        )

    with col3:

        st.subheader("🛡️ Investigation Support")

        st.write(
            """
            High-risk transactions can be flagged for further
            investigation by financial analysts.
            """
        )

    st.divider()

    st.header(
        "How the System Works"
    )

    process_col1, process_col2 = st.columns(2)

    with process_col1:

        st.markdown(
            """
            **1. Transaction Input**

            Financial transaction information is entered.

            **2. Feature Analysis**

            Transaction and account behaviour is analysed.

            **3. ML Prediction**

            The model estimates fraud probability.
            """
        )

    with process_col2:

        st.markdown(
            """
            **4. Risk Score**

            Probability is converted into a risk score.

            **5. Risk Category**

            The account/transaction is categorized.

            **6. Investigation**

            Suspicious cases can be investigated further.
            """
        )


# =============================================================================
# TRANSACTION ANALYSIS TAB
# =============================================================================

with analysis_tab:

    st.header(
        "🔍 Transaction Analysis"
    )

    st.write(
        "Enter the transaction information below."
    )

    # -------------------------------------------------------------------------
    # TRANSACTION DETAILS
    # -------------------------------------------------------------------------

    st.subheader(
        "Transaction Details"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        transaction_type = st.selectbox(
            "Transaction Type",
            [
                "PAYMENT",
                "TRANSFER",
                "CASH_OUT",
                "DEBIT",
                "CASH_IN"
            ]
        )

    with col2:

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=1000.0,
            step=100.0
        )

    with col3:

        transaction_hour = st.slider(
            "Transaction Hour",
            min_value=0,
            max_value=23,
            value=12
        )

    # -------------------------------------------------------------------------
    # SENDER DETAILS
    # -------------------------------------------------------------------------

    st.subheader(
        "Sender / Origin Account"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        sender_old_balance = st.number_input(
            "Sender Previous Balance",
            min_value=0.0,
            value=10000.0,
            step=500.0
        )

    with col2:

        sender_new_balance = st.number_input(
            "Sender New Balance",
            min_value=0.0,
            value=9000.0,
            step=500.0
        )

    with col3:

        sender_transaction_count = st.number_input(
            "Sender Transaction Count",
            min_value=0,
            value=5,
            step=1
        )

    # -------------------------------------------------------------------------
    # RECEIVER DETAILS
    # -------------------------------------------------------------------------

    st.subheader(
        "Receiver / Destination Account"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        receiver_old_balance = st.number_input(
            "Receiver Previous Balance",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

    with col2:

        receiver_new_balance = st.number_input(
            "Receiver New Balance",
            min_value=0.0,
            value=6000.0,
            step=500.0
        )

    with col3:

        receiver_transaction_count = st.number_input(
            "Receiver Transaction Count",
            min_value=0,
            value=5,
            step=1
        )

    # -------------------------------------------------------------------------
    # BEHAVIOURAL INFORMATION
    # -------------------------------------------------------------------------

    st.subheader(
        "Account Behaviour"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        unique_counterparties = st.number_input(
            "Unique Counterparties",
            min_value=0,
            value=3,
            step=1
        )

    with col2:

        suspicious_history = st.number_input(
            "Previous Suspicious Transactions",
            min_value=0,
            value=0,
            step=1
        )

    with col3:

        account_activity_days = st.number_input(
            "Account Activity Days",
            min_value=0,
            value=100,
            step=1
        )

    st.divider()

    # -------------------------------------------------------------------------
    # ANALYZE BUTTON
    # -------------------------------------------------------------------------

    analyze_button = st.button(
        "🔍 ANALYZE TRANSACTION",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        with st.spinner(
            "Analyzing transaction behaviour..."
        ):

            time.sleep(1)

        result = generate_demo_prediction()

        prediction = result["prediction"]

        probability = result["probability"]

        risk_score = result["risk_score"]

        risk_category = result["risk_category"]

        transaction_id = generate_transaction_id()

        # ---------------------------------------------------------------------
        # UPDATE STATISTICS
        # ---------------------------------------------------------------------

        st.session_state.analysis_count += 1

        if prediction == "FRAUD":

            st.session_state.fraud_count += 1

        else:

            st.session_state.not_fraud_count += 1

        st.session_state.last_result = result

        # ---------------------------------------------------------------------
        # RESULT HEADER
        # ---------------------------------------------------------------------

        st.divider()

        st.header(
            "🚨 Detection Result"
        )

        result_col1, result_col2 = st.columns(
            [1.2, 1]
        )

        # ---------------------------------------------------------------------
        # GAUGE
        # ---------------------------------------------------------------------

        with result_col1:

            st.plotly_chart(
                create_gauge(
                    probability
                ),
                use_container_width=True
            )

        # ---------------------------------------------------------------------
        # RESULT
        # ---------------------------------------------------------------------

        with result_col2:

            st.metric(
                "Prediction",
                prediction
            )

            st.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )

            st.metric(
                "Risk Score",
                f"{risk_score:.2f}/100"
            )

            st.metric(
                "Risk Category",
                risk_category
            )

        # ---------------------------------------------------------------------
        # STATUS
        # ---------------------------------------------------------------------

        if prediction == "FRAUD":

            st.error(
                "🚨 Potential FRAUD detected. Further investigation is recommended."
            )

        else:

            st.success(
                "✅ Transaction classified as NOT FRAUD."
            )

        st.info(
            get_risk_description(
                risk_category
            )
        )

        # ---------------------------------------------------------------------
        # TRANSACTION SUMMARY
        # ---------------------------------------------------------------------

        st.subheader(
            "Transaction Summary"
        )

        summary = pd.DataFrame({

            "Parameter": [

                "Transaction ID",

                "Transaction Type",

                "Transaction Amount",

                "Transaction Hour",

                "Sender Previous Balance",

                "Sender New Balance",

                "Receiver Previous Balance",

                "Receiver New Balance",

                "Sender Transaction Count",

                "Receiver Transaction Count",

                "Unique Counterparties",

                "Previous Suspicious Transactions",

                "Account Activity Days"

            ],

            "Value": [

                transaction_id,

                transaction_type,

                f"₹ {amount:,.2f}",

                transaction_hour,

                f"₹ {sender_old_balance:,.2f}",

                f"₹ {sender_new_balance:,.2f}",

                f"₹ {receiver_old_balance:,.2f}",

                f"₹ {receiver_new_balance:,.2f}",

                sender_transaction_count,

                receiver_transaction_count,

                unique_counterparties,

                suspicious_history,

                account_activity_days

            ]

        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        # ---------------------------------------------------------------------
        # RECOMMENDATION
        # ---------------------------------------------------------------------

        st.subheader(
            "Recommended Action"
        )

        if prediction == "FRAUD":

            st.warning(
                """
                Priority review recommended.

                Suggested investigation:

                • Review account transaction history

                • Check unusual transaction velocity

                • Verify sender/receiver relationship

                • Review previous suspicious activity

                • Perform appropriate AML/KYC checks
                """
            )

        else:

            st.success(
                """
                No immediate suspicious signal was generated
                in this demonstration. Continue normal monitoring.
                """
            )

        # ---------------------------------------------------------------------
        # DOWNLOAD REPORT
        # ---------------------------------------------------------------------

        report = pd.DataFrame({

            "Transaction ID": [transaction_id],

            "Transaction Type": [transaction_type],

            "Amount": [amount],

            "Transaction Hour": [transaction_hour],

            "Fraud Probability": [probability],

            "Risk Score": [risk_score],

            "Risk Category": [risk_category],

            "Prediction": [prediction]

        })

        csv_data = report.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.download_button(
            "⬇️ Download Detection Report",
            data=csv_data,
            file_name="fraud_detection_report.csv",
            mime="text/csv",
            use_container_width=True
        )


# =============================================================================
# MODEL PERFORMANCE TAB
# =============================================================================

with performance_tab:

    st.header(
        "📊 Model Performance"
    )

    st.warning(
        """
        DEMO METRICS ONLY

        These values are illustrative and are NOT actual results
        calculated from the trained dataset.
        """
    )

    # -------------------------------------------------------------------------
    # MODEL SELECTION
    # -------------------------------------------------------------------------

    selected_model = st.selectbox(
        "Select Model",
        list(
            DEMO_METRICS.keys()
        )
    )

    metrics = DEMO_METRICS[
        selected_model
    ]

    # -------------------------------------------------------------------------
    # METRIC CARDS
    # -------------------------------------------------------------------------

    st.subheader(
        f"{selected_model} Performance"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Accuracy",
            f"{metrics['Accuracy']:.2f}%"
        )

    with col2:

        st.metric(
            "Precision",
            f"{metrics['Precision']:.2f}%"
        )

    with col3:

        st.metric(
            "Recall",
            f"{metrics['Recall']:.2f}%"
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "F1 Score",
            f"{metrics['F1 Score']:.2f}%"
        )

    with col2:

        st.metric(
            "ROC-AUC",
            f"{metrics['ROC-AUC']:.2f}%"
        )

    with col3:

        st.metric(
            "PR-AUC",
            f"{metrics['PR-AUC']:.2f}%"
        )

    # -------------------------------------------------------------------------
    # COMPARISON TABLE
    # -------------------------------------------------------------------------

    st.subheader(
        "Model Comparison"
    )

    comparison_rows = []

    for model_name, values in DEMO_METRICS.items():

        comparison_rows.append({

            "Model": model_name,

            "Accuracy":
                f"{values['Accuracy']:.2f}%",

            "Precision":
                f"{values['Precision']:.2f}%",

            "Recall":
                f"{values['Recall']:.2f}%",

            "F1 Score":
                f"{values['F1 Score']:.2f}%",

            "ROC-AUC":
                f"{values['ROC-AUC']:.2f}%",

            "PR-AUC":
                f"{values['PR-AUC']:.2f}%"

        })

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    # -------------------------------------------------------------------------
    # BEST MODEL
    # -------------------------------------------------------------------------

    best_model = max(
        DEMO_METRICS.items(),
        key=lambda x:
        x[1]["PR-AUC"]
    )

    st.success(
        f"Best demonstration model based on PR-AUC: {best_model[0]}"
    )

    st.info(
        """
        For the final project, these metrics must be replaced with
        genuine test-set results from your trained models.
        """
    )


# =============================================================================
# RISK INTELLIGENCE TAB
# =============================================================================

with intelligence_tab:

    st.header(
        "🛡️ Risk Intelligence"
    )

    st.write(
        """
        The following risk categories are used by the demonstration
        interface.
        """
    )

    risk_table = pd.DataFrame({

        "Risk Category": [

            "LOW RISK",

            "MEDIUM RISK",

            "HIGH RISK",

            "CRITICAL RISK"

        ],

        "Probability": [

            "0% – 20%",

            "20% – 50%",

            "50% – 75%",

            "75% – 100%"

        ],

        "Recommended Action": [

            "Normal Monitoring",

            "Monitor",

            "Investigate",

            "Priority Investigation"

        ]

    })

    st.dataframe(
        risk_table,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.header(
        "Potential Mule Account Indicators"
    )

    indicators = [
        "High transaction velocity",
        "Large amounts moving through an account",
        "Large number of counterparties",
        "Unusual incoming/outgoing ratio",
        "Repeated high-value transfers",
        "Sudden changes in account activity",
        "Rapid movement of received funds",
        "Unusual transaction timing",
        "Repeated suspicious transaction patterns"
    ]

    for indicator in indicators:

        st.write(
            "• " + indicator
        )

    st.info(
        """
        These indicators represent risk signals only.
        A risk prediction should not automatically be interpreted
        as proof of criminal activity.
        """
    )


# =============================================================================
# ABOUT PROJECT TAB
# =============================================================================

with about_tab:

    st.header(
        "ℹ️ About MuleGuard AI"
    )

    st.subheader(
        "Problem Statement"
    )

    st.write(
        """
        AI/ML-Based Classification of Suspicious Mule Accounts
        """
    )

    st.write(
        """
        Financial institutions process millions of transactions.
        Some accounts may be used to receive, transfer, or move
        illicit funds.

        The objective of this project is to use Machine Learning
        and behavioural analysis to identify potentially suspicious
        transactions and accounts.
        """
    )

    st.subheader(
        "Project Pipeline"
    )

    pipeline = pd.DataFrame({

        "Stage": [

            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8

        ],

        "Process": [

            "Data Collection",

            "Data Exploration",

            "Data Preprocessing",

            "Feature Engineering",

            "Model Training",

            "Model Evaluation",

            "Risk Scoring",

            "Streamlit Deployment"

        ]

    })

    st.dataframe(
        pipeline,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Machine Learning Models"
    )

    st.write(
        """
        The complete ML project can use:

        • Logistic Regression

        • Random Forest

        • XGBoost

        • Hyperparameter Optimization

        • Threshold Optimization

        • Feature Importance

        • Model Explainability
        """
    )

    st.subheader(
        "Future Scope"
    )

    future_scope = [

        "Real-time transaction monitoring",

        "Graph Neural Networks",

        "Graph-based mule account detection",

        "Anomaly Detection",

        "Autoencoders",

        "SHAP Explainability",

        "Real-time banking integration",

        "AML rule engine integration",

        "Concept drift monitoring",

        "Investigator feedback system"

    ]

    for item in future_scope:

        st.write(
            "• " + item
        )

    st.divider()

    st.warning(
        """
        DEMO DISCLAIMER

        This version of the application does not use the trained
        Machine Learning model.

        Predictions are randomly generated for demonstrating
        the Streamlit interface.

        Performance numbers shown on the Model Performance page
        are illustrative only.
        """
    )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "🛡️ MuleGuard AI | AI/ML-Based Fraud & Suspicious Mule Account Detection"
)

st.caption(
    "DEMO VERSION — Randomized predictions for interface demonstration only."
)