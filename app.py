import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import json
import joblib
import tensorflow as tf
import yfinance as yf
import pandas as pd

# =========================
# 🧠 CUSTOM ATTENTION LAYER
# =========================
class Attention(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1))
        self.b = self.add_weight(shape=(input_shape[1], 1))

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

# =========================
# LOAD ALL FILES
# =========================
@st.cache_resource
def load_all():

    # ✅ LOAD FULL MODEL (.keras)
    model = tf.keras.models.load_model(
        "model/final_model.keras",
        custom_objects={"Attention": Attention},
        compile=False
    )

    scaler_X = joblib.load("model/scaler_X.pkl")
    scaler_y = joblib.load("model/scaler_y.pkl")

    with open("model/stocks.json") as f:
        STOCKS = json.load(f)

    with open("model/feature_columns.json") as f:
        FEATURE_COLUMNS = json.load(f)

    with open("model/config.json") as f:
        CONFIG = json.load(f)
        SEQ_LEN = CONFIG.get("SEQ_LEN", 20)

    return model, scaler_X, scaler_y, STOCKS, FEATURE_COLUMNS, SEQ_LEN


model, scaler_X, scaler_y, STOCKS, FEATURE_COLUMNS, SEQ_LEN = load_all()

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Stock AI", layout="wide")

st.title("📈 Multi-Stock AI Prediction Dashboard")
st.caption("LSTM + Attention | Multi-Stock Model")

# 👉 USER SELECTS WHAT TO DISPLAY (NOT MODEL INPUT)
selected_stocks = st.multiselect(
    "📊 Select Stocks to View",
    STOCKS,
    default=STOCKS[:3]
)

# =========================
# PREDICTION FUNCTION
# =========================
def predict_all():

    # ⚠️ ALWAYS use FULL STOCK LIST
    df = yf.download(STOCKS, period="60d")["Close"].dropna()

    df_feat = df.copy()

    for s in STOCKS:
        df_feat[f"{s}_RET"] = df_feat[s].pct_change()
        df_feat[f"{s}_MA7"] = df_feat[s].rolling(7).mean()
        df_feat[f"{s}_MA21"] = df_feat[s].rolling(21).mean()
        df_feat[f"{s}_STD"] = df_feat[s].rolling(21).std()
        df_feat[f"{s}_MOM"] = df_feat[s] - df_feat[s].shift(5)
        df_feat[f"{s}_ROC"] = df_feat[s].pct_change(5)
        df_feat[f"{s}_SENT"] = 0  # no API (safe)

    df_feat = df_feat.dropna()

    # ensure feature consistency
    for col in FEATURE_COLUMNS:
        if col not in df_feat.columns:
            df_feat[col] = 0

    df_feat = df_feat[FEATURE_COLUMNS]

    X = scaler_X.transform(df_feat)
    X = X[-SEQ_LEN:].reshape(1, SEQ_LEN, X.shape[1])

    pred_reg, pred_cls = model.predict(X, verbose=0)

    pred = scaler_y.inverse_transform(pred_reg)[0]

    # 🔥 SAME LOGIC AS NOTEBOOK
    pred = pred - np.mean(pred)
    pred = 0.7 * pred + 0.3 * np.mean(pred)
    pred = np.clip(pred, -0.08, 0.08)

    if np.std(pred) > 0:
        pred = pred / np.max(np.abs(pred)) * 0.05

    last_price = df.iloc[-1].values
    next_price = last_price * (1 + pred)

    return last_price, pred, pred_cls[0], next_price


# =========================
# RUN PREDICTION
# =========================
if st.button("🚀 Run Prediction"):

    last_price, pred, cls, next_price = predict_all()

    st.subheader("📊 Predictions")

    for s in selected_stocks:
        idx = STOCKS.index(s)

        col1, col2, col3 = st.columns(3)

        col1.metric("💰 Current Price", f"{last_price[idx]:.2f}")
        col2.metric("📈 Return", f"{pred[idx]*100:.2f}%")
        col3.metric("🔮 Next Price", f"{next_price[idx]:.2f}")

        # SIGNAL LOGIC
        if pred[idx] > 0.02:
            signal = "STRONG BUY"
        elif pred[idx] > 0.005:
            signal = "BUY"
        elif pred[idx] < -0.02:
            signal = "STRONG SELL"
        elif pred[idx] < -0.005:
            signal = "SELL"
        else:
            signal = "HOLD"

        confidence = min(abs(pred[idx]) * 12 + abs(cls[idx] - 0.5), 0.9)

        st.write(f"📢 Signal: **{signal}**")
        st.progress(float(confidence))

        # 📉 CHART
        hist = yf.download(s, period="60d")
        st.line_chart(hist["Close"])

        st.divider()
