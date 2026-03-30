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
# 📦 LOAD MODEL + FILES
# =========================
@st.cache_resource
def load_all():
    model = tf.keras.models.load_model(
        "model/final_model.keras",
        custom_objects={"Attention": Attention},
        compile=False,
        safe_mode=False
    )

    scaler_X = joblib.load("model/scaler_X.pkl")
    scaler_y = joblib.load("model/scaler_y.pkl")

    with open("model/stocks.json") as f:
        STOCKS = json.load(f)

    with open("model/feature_columns.json") as f:
        FEATURE_COLUMNS = json.load(f)

    try:
        with open("model/config.json") as f:
            CONFIG = json.load(f)
            SEQ_LEN = CONFIG.get("SEQ_LEN", 20)
    except:
        SEQ_LEN = 20

    return model, scaler_X, scaler_y, STOCKS, FEATURE_COLUMNS, SEQ_LEN

model, scaler_X, scaler_y, STOCKS, FEATURE_COLUMNS, SEQ_LEN = load_all()

# =========================
# 📊 SIGNAL FUNCTION (same as notebook)
# =========================
def generate_signals(pred, cls):
    signals = []
    confidence = []

    for r, c in zip(pred, cls):

        conf = min(abs(r) * 12 + abs(c - 0.5), 0.9)
        confidence.append(conf)

        if r > 0.02:
            signals.append("STRONG BUY")
        elif r > 0.005:
            signals.append("BUY")
        elif r < -0.02:
            signals.append("STRONG SELL")
        elif r < -0.005:
            signals.append("SELL")
        else:
            signals.append("HOLD")

    return signals, confidence

# =========================
# 🎨 UI
# =========================
st.set_page_config(page_title="Stock AI", layout="wide")
st.title("📈 Multi-Stock Prediction Dashboard")

# =========================
# 🚀 PREDICTION
# =========================
if st.button("Predict Now 🚀"):

    with st.spinner("Fetching data & predicting..."):

        try:
            df = yf.download(STOCKS, period="60d")["Close"].dropna()
        except:
            st.error("Failed to fetch data")
            st.stop()

        if df.empty:
            st.error("No data available")
            st.stop()

        df_feat = df.copy()

        # =========================
        # FEATURE ENGINEERING (same as notebook)
        # =========================
        for s in STOCKS:
            df_feat[f"{s}_RET"] = df_feat[s].pct_change()
            df_feat[f"{s}_MA7"] = df_feat[s].rolling(7).mean()
            df_feat[f"{s}_MA21"] = df_feat[s].rolling(21).mean()
            df_feat[f"{s}_STD"] = df_feat[s].rolling(21).std()
            df_feat[f"{s}_MOM"] = df_feat[s] - df_feat[s].shift(5)
            df_feat[f"{s}_ROC"] = df_feat[s].pct_change(5)

            # same as notebook
            df_feat[f"{s}_SENT"] = 0

        df_feat = df_feat.dropna()

        # =========================
        # FEATURE ALIGNMENT
        # =========================
        for col in FEATURE_COLUMNS:
            if col not in df_feat.columns:
                df_feat[col] = 0

        df_feat = df_feat[FEATURE_COLUMNS]

        # =========================
        # SCALING + SEQUENCE
        # =========================
        X = scaler_X.transform(df_feat)
        X = X[-SEQ_LEN:].reshape(1, SEQ_LEN, X.shape[1])

        # =========================
        # MODEL PREDICTION
        # =========================
        pred_reg, pred_cls = model.predict(X, verbose=0)

        pred = scaler_y.inverse_transform(pred_reg)[0]

        # =========================
        # 🔥 EXACT NOTEBOOK LOGIC
        # =========================
        pred = pred - np.mean(pred)
        pred = 0.7 * pred + 0.3 * np.mean(pred)
        pred = np.clip(pred, -0.08, 0.08)

        if np.std(pred) > 0:
            pred = pred / np.max(np.abs(pred)) * 0.05

        # =========================
        # FINAL PRICES
        # =========================
        last_price = df.iloc[-1].values
        next_price = last_price * (1 + pred)

        # =========================
        # SIGNALS
        # =========================
        signals, confidence = generate_signals(pred, pred_cls[0])

        # =========================
        # DISPLAY
        # =========================
        st.subheader("📊 Predictions")

        cols = st.columns(len(STOCKS))

        for i, s in enumerate(STOCKS):
            with cols[i]:
                st.metric(
                    label=s,
                    value=f"{next_price[i]:.2f}",
                    delta=f"{pred[i]*100:.2f}%"
                )
                st.write(f"Signal: {signals[i]}")
                st.write(f"Confidence: {confidence[i]:.2f}")
