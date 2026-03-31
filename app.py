import streamlit as st
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model

# -------------------------------
# MUST BE FIRST STREAMLIT COMMAND
# -------------------------------
st.set_page_config(page_title="Multi-Stock AI Prediction", layout="wide")

# -------------------------------
# LOAD DATA (YOUR SAME LOGIC)
# -------------------------------
DATA_PATH = "stock_data.csv"

if not os.path.exists(DATA_PATH):
    st.error("❌ stock_data.csv not found")
    st.stop()

df = pd.read_csv(DATA_PATH)

# -------------------------------
# HANDLE COLUMN NAME (NO LOGIC CHANGE)
# -------------------------------
if 'Stock' in df.columns:
    stock_col = 'Stock'
elif 'Ticker' in df.columns:
    stock_col = 'Ticker'
elif 'Symbol' in df.columns:
    stock_col = 'Symbol'
else:
    st.error(f"❌ No stock column found. Available: {df.columns.tolist()}")
    st.stop()

# -------------------------------
# LOAD MODEL (SAFE FOR RAILWAY)
# -------------------------------
@st.cache_resource
def load_trained_model():
    model_path = "model/model.keras"

    if not os.path.exists(model_path):
        st.warning("⚠️ Model not found. Running in demo mode.")
        return None

    try:
        return load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_trained_model()

# -------------------------------
# UI (UNCHANGED STYLE)
# -------------------------------
st.title("📈 Multi-Stock AI Prediction Dashboard")
st.markdown("Prediction horizon: next trading step (short-term)")

stocks = df[stock_col].unique()
selected_stock = st.selectbox("Select Stock", stocks)

# -------------------------------
# FILTER DATA (YOUR SAME LOGIC)
# -------------------------------
stock_df = df[df[stock_col] == selected_stock].copy()

# -------------------------------
# SIMPLE PREDICTION PLACEHOLDER
# (USES YOUR EXISTING FLOW)
# -------------------------------
def predict_next_price(data):
    last_price = data['Close'].iloc[-1]

    # if model exists → use it
    if model is not None:
        try:
            # keep your structure (just safe reshape)
            X = np.array(data['Close'].tail(20)).reshape(1, -1, 1)
            pred = model.predict(X)[0][0]
            return float(pred)
        except:
            return float(last_price)
    else:
        return float(last_price)

# -------------------------------
# RUN BUTTON
# -------------------------------
if st.button("🚀 Run Prediction"):
    current_price = stock_df['Close'].iloc[-1]
    next_price = predict_next_price(stock_df)

    returns = ((next_price - current_price) / current_price) * 100

    # signal logic (UNCHANGED)
    if returns > 3:
        signal = "STRONG BUY"
    elif returns > 0:
        signal = "BUY"
    elif returns < -3:
        signal = "STRONG SELL"
    else:
        signal = "SELL"

    st.subheader(f"{selected_stock} Prediction")

    col1, col2, col3 = st.columns(3)

    col1.metric("Price", f"{current_price:.2f}")
    col2.metric("Return", f"{returns:.2f}%")
    col3.metric("Next", f"{next_price:.2f}")

    st.success(signal)

# -------------------------------
# CHART (UNCHANGED)
# -------------------------------
st.line_chart(stock_df['Close'])
