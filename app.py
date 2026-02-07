import streamlit as st
import numpy as np
import joblib
import pandas as pd
import time
import matplotlib.pyplot as plt
from src.shap_utils import explain_prediction
from src.train_utils import train_model
from src.data_preprocessing import preprocess_single_sample, get_bulk_samples


if "show_explanation" not in st.session_state:
    st.session_state.show_explanation = False

if "show_distribution" not in st.session_state:
    st.session_state.show_distribution = False

if "distribution_seed" not in st.session_state:
    st.session_state.distribution_seed = 0


model = joblib.load("models/fraud_model.pkl")


st.header("🔧 Model Training")

if st.button("🚀 Train / Retrain Model"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(percent, message):
        progress_bar.progress(percent)
        status_text.text(message)
        time.sleep(0.5)

    with st.spinner("Training in progress..."):
        train_model(progress_callback=update_progress)

    st.success("✅ Model training completed!")

st.title("Cyber Fraud Detection System")

row_index = st.number_input(
    "Transaction index",
    min_value=0,
    max_value=1000,
    value=0
)

if st.button("Check Transaction"):
    data = preprocess_single_sample(row_index)
    st.session_state.last_sample = data
    st.session_state.distribution_seed += 1
    st.session_state.show_explanation = True
    st.session_state.show_distribution = True

    prob = model.predict_proba(data)[0][1]
    risk_score = round(prob * 100, 2)

    st.info(f"⚠️ Fraud Risk Score: {risk_score}%")

    if risk_score < 10:
        decision = "✅ Approved"
        color = "green"
    elif risk_score < 40:
        decision = "⚠️ Review Required"
        color = "orange"
    else:
        decision = "⛔ Block Transaction"
        color = "red"

    st.markdown(
        f"<h3 style='color:{color};'>{decision}</h3>",
        unsafe_allow_html=True
    )


st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if st.button("Explain Prediction"):
        st.session_state.show_explanation = True

with col2:
    if st.button("Show Risk Distribution"):
        st.session_state.show_distribution = True

if st.session_state.show_explanation:
    st.subheader("🔍 Why this score?")

    if "last_sample" not in st.session_state:
        st.warning("⚠️ Please check a transaction first.")
    else:
        import numpy as np

        data = st.session_state.last_sample
        explainer, shap_values = explain_prediction(data)

        if isinstance(shap_values, list):
            values = np.array(shap_values[1]).flatten()
        else:
            values = np.array(shap_values).flatten()

        feature_names = data.columns.to_numpy()
        max_len = min(len(values), len(feature_names))
        values = values[:max_len]
        feature_names = feature_names[:max_len]
        top_idx = np.argsort(np.abs(values))[-5:][::-1]

        st.write("Top contributing features:")
        for f in feature_names[top_idx]:
            st.write(f"• {f}")


if st.session_state.show_distribution:
    st.subheader("📊 Fraud Risk Distribution")

    with st.spinner("Computing risk distribution..."):
        samples = get_bulk_samples(
            n=300,
            random_state=st.session_state.distribution_seed
        )

        probs = model.predict_proba(samples)[:, 1] * 100

        fig, ax = plt.subplots()
        ax.hist(probs, bins=20, alpha=0.7)
        ax.set_xlabel("Fraud Risk (%)")
        ax.set_ylabel("Number of Transactions")
        ax.set_title("Fraud Risk Distribution")
        if "last_sample" in st.session_state:
            selected_prob = (
                model.predict_proba(st.session_state.last_sample)[0][1] * 100
            )
            ax.axvline(
                selected_prob,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Selected Transaction"
            )
            ax.legend()

        st.pyplot(fig)
