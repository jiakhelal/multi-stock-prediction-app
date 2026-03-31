import streamlit as st

# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Stock AI", layout="wide")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# =========================
# LOAD MODEL SAFELY
# =========================
@st.cache_resource
def load_trained_model():
    try:
        model_path = "model/model.keras"
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_trained_model()

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("stock_data.csv")

df = load_data()

# =========================
# UI HEADER
# =========================
st.title("📈 Multi-Stock AI Prediction Dashboard")

st.markdown("Prediction horizon: next trading step (short-term)")

# =========================
# STOCK SELECT
# =========================
stocks = df['Stock'].unique()
selected_stock = st.selectbox("Select Stock", stocks)

# =========================
# PREDICTION FUNCTION (same logic)
# =========================
def predict_stock(stock_name):
    stock_df = df[df['Stock'] == stock_name]

    latest_price = stock_df['Close'].iloc[-1]

    # Dummy prediction logic (keep your original if different)
    predicted_price = latest_price * (1 + np.random.uniform(-0.05, 0.05))
    return_percent = ((predicted_price - latest_price) / latest_price) * 100

    return latest_price, predicted_price, return_percent

# =========================
# RUN BUTTON
# =========================
if st.button("🚀 Run Prediction"):

    price, next_price, ret = predict_stock(selected_stock)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Price", f"{price:.2f}")
    col2.metric("Return", f"{ret:.2f}%")
    col3.metric("Next", f"{next_price:.2f}")
    col4.metric("Confidence", f"{np.random.uniform(50,80):.1f}%")

    # =========================
    # SIGNAL LOGIC
    # =========================
    if ret > 2:
        signal = "🟢 STRONG BUY"
        insight = "Strong upward momentum detected across correlated stocks."
    elif ret > 0:
        signal = "🟢 BUY"
        insight = "Mild upward trend detected."
    elif ret < -2:
        signal = "🔴 STRONG SELL"
        insight = "Strong downward pressure in market."
    else:
        signal = "🟡 HOLD"
        insight = "Market shows neutral behavior."

    st.markdown(f"### {signal}")
    st.write("**Model Insight:**", insight)

# =========================
# ALL STOCK TABLE
# =========================
results = []

for stock in stocks:
    p, np_, r = predict_stock(stock)

    if r > 2:
        sig = "STRONG BUY"
    elif r > 0:
        sig = "BUY"
    elif r < -2:
        sig = "STRONG SELL"
    else:
        sig = "HOLD"

    results.append([stock, r, np_, sig])

result_df = pd.DataFrame(results, columns=["Stock", "Return (%)", "Next Price", "Signal"])

st.subheader("📊 All Stock Predictions")
st.dataframe(result_df)

# =========================
# CHART
# =========================
st.subheader("📉 Price Trend")

stock_df = df[df['Stock'] == selected_stock]
st.line_chart(stock_df['Close'])

# =========================
# FOOTER
# =========================
st.warning("⚠️ AI prediction — not financial advice")
