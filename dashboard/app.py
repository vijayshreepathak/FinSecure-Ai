import os
import time
from typing import Dict, Any, List
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="FinSecure AI Fraud Monitoring", layout="wide")

st.title("FinSecure AI — Real-Time Fraud Detection Dashboard")

threshold_col, _ = st.columns([1, 3])
with threshold_col:
    ui_threshold = st.slider("Alert Threshold (Risk Score)", min_value=0.1, max_value=0.99, value=0.6, step=0.01)


def call_api_predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def call_api_metrics() -> Dict[str, Any]:
    resp = requests.get(f"{API_URL}/metrics", timeout=10)
    resp.raise_for_status()
    return resp.json()


def call_api_alerts() -> List[Dict[str, Any]]:
    resp = requests.get(f"{API_URL}/alerts", timeout=10)
    resp.raise_for_status()
    return resp.json()


def call_api_predict_batch_file(file) -> Dict[str, Any]:
    files = {"file": (file.name, file.getvalue(), "text/csv")}
    resp = requests.post(f"{API_URL}/predict_batch", files=files, timeout=60)
    resp.raise_for_status()
    return resp.json()


st.subheader("Transaction Input")

input_cols1 = st.columns(3)
with input_cols1[0]:
    time_val = st.number_input("Time", min_value=0.0, value=0.0)
with input_cols1[1]:
    amount_val = st.number_input("Amount", min_value=0.0, value=1.0, step=0.1)
with input_cols1[2]:
    has_label = st.checkbox("Provide True Label")
true_label_val = None
if has_label:
    true_label_val = st.selectbox("True Class Label", options=[0, 1], index=0)

v_values: Dict[str, float] = {}
with st.expander("Advanced PCA Features V1–V28"):
    v_cols = []
    for i in range(1, 29):
        v_cols.append(st.number_input(f"V{i}", value=0.0, key=f"V{i}"))
    for i in range(1, 29):
        v_values[f"V{i}"] = float(v_cols[i - 1])

if st.button("Submit Transaction"):
    payload: Dict[str, Any] = {"Time": float(time_val), "Amount": float(amount_val)}
    for i in range(1, 29):
        payload[f"V{i}"] = v_values[f"V{i}"]
    if true_label_val is not None:
        payload["Class"] = int(true_label_val)
    try:
        result = call_api_predict(payload)
        st.session_state["last_prediction"] = {"input": payload, "output": result}
    except Exception as e:
        st.error(f"Prediction failed: {e}")

if "last_prediction" in st.session_state:
    st.subheader("Last Prediction")
    out = st.session_state["last_prediction"]["output"]
    ui_label = 1 if out["risk_score"] >= ui_threshold else 0
    cols = st.columns(4)
    cols[0].metric("Fraud Probability", f"{out['probability']:.4f}")
    cols[1].metric("Risk Score", f"{out['risk_score']:.4f}")
    cols[2].metric("Model Label", str(out["label"]))
    cols[3].metric("UI Label (Threshold)", str(ui_label))
    st.markdown("**Top Contributing Features**")
    if out["explanations"]:
        df_exp = pd.DataFrame(out["explanations"])
        st.dataframe(df_exp)
    else:
        st.write("No explanation available")

st.subheader("Bulk Ingestion Simulation")

upload_col1, upload_col2 = st.columns([2, 1])
with upload_col1:
    uploaded_file = st.file_uploader("Upload CSV of Transactions", type=["csv"])
with upload_col2:
    if uploaded_file is not None:
        if st.button("Run Batch Inference"):
            progress = st.progress(0)
            status_placeholder = st.empty()
            try:
                status_placeholder.write("Uploading and scoring batch")
                result = call_api_predict_batch_file(uploaded_file)
                total = result.get("count", 0)
                progress.progress(100)
                status_placeholder.write(f"Completed batch scoring for {total} records")
                st.session_state["last_batch_result"] = result
            except Exception as e:
                status_placeholder.write(f"Batch inference failed: {e}")
                progress.progress(0)

if "last_batch_result" in st.session_state:
    st.markdown("**Last Batch Summary**")
    batch = st.session_state["last_batch_result"]
    st.write(f"Records scored: {batch['count']}")

st.subheader("Fraud Monitoring and Metrics")

metrics_data = None
alerts_data = None
try:
    metrics_data = call_api_metrics()
    alerts_data = call_api_alerts()
except Exception as e:
    st.warning(f"Could not fetch metrics or alerts: {e}")

if metrics_data is not None:
    top_cols = st.columns(5)
    top_cols[0].metric("Total Transactions", str(metrics_data["total"]))
    top_cols[1].metric("Fraud Count", str(metrics_data["fraud_count"]))
    top_cols[2].metric("Fraud Rate", f"{metrics_data['fraud_rate']*100:.4f}%")
    if metrics_data["precision"] is not None:
        top_cols[3].metric("Precision", f"{metrics_data['precision']:.4f}")
    else:
        top_cols[3].metric("Precision", "N/A")
    if metrics_data["recall"] is not None:
        top_cols[4].metric("Recall", f"{metrics_data['recall']:.4f}")
    else:
        top_cols[4].metric("Recall", "N/A")
    plot_cols = st.columns(4)
    fraud_pie_df = pd.DataFrame(
        {
            "label": ["Fraud", "Non-Fraud"],
            "count": [metrics_data["fraud_count"], metrics_data["total"] - metrics_data["fraud_count"]],
        }
    )
    with plot_cols[0]:
        fig_pie = px.pie(fraud_pie_df, names="label", values="count", hole=0.4, title="Fraud vs Non-Fraud Share")
        st.plotly_chart(fig_pie, use_container_width=True)
    by_hour = metrics_data.get("by_hour", [])
    if by_hour:
        df_hour = pd.DataFrame(by_hour)
        df_hour_melt = df_hour.melt(id_vars="hour", value_vars=["fraud", "non_fraud"], var_name="type", value_name="count")
        with plot_cols[1]:
            fig_hour = px.line(df_hour_melt, x="hour", y="count", color="type", title="Fraud Count by Hour")
            st.plotly_chart(fig_hour, use_container_width=True)
    amount_hist = metrics_data.get("amount_hist", [])
    if amount_hist:
        df_amt = pd.DataFrame(amount_hist)
        df_amt_melt = df_amt.melt(id_vars="bin", value_vars=["fraud", "non_fraud"], var_name="type", value_name="count")
        with plot_cols[2]:
            fig_amt = px.bar(df_amt_melt, x="bin", y="count", color="type", title="Fraud by Amount Bin")
            fig_amt.update_layout(xaxis_title="Amount Bin", yaxis_title="Count")
            st.plotly_chart(fig_amt, use_container_width=True)
    if metrics_data["tp"] + metrics_data["fn"] + metrics_data["fp"] + metrics_data["tn"] > 0:
        cm_array = np.array(
            [
                [metrics_data["tn"], metrics_data["fp"]],
                [metrics_data["fn"], metrics_data["tp"]],
            ]
        )
        cm_df = pd.DataFrame(cm_array, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
        with plot_cols[3]:
            fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)

st.subheader("Live Fraud Alerts")

if alerts_data is not None and alerts_data:
    alerts_df = pd.DataFrame(alerts_data)
    st.dataframe(alerts_df)
else:
    st.write("No alerts yet")

time.sleep(2.0)
try:
    st.rerun()
except Exception:
    pass


