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
# 🧠 CUSTOM ATTENTION LAYER (FIXED INDENTATION)
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
# 🧠 BUILD MODEL ARCHITECTURE (IMPORTANT)
# =========================
def build_model(input_shape, output_dim):

    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    x = Attention()(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)

    out_reg = tf.keras.layers.Dense(output_dim, name="regression")(x)
    out_cls = tf.keras.layers.Dense(output_dim, activation="sigmoid", name="classification")(x)

    model = tf.keras.Model(inputs, [out_reg, out_cls])

    return model


# =========================
# 📂 LOAD EVERYTHING
# =========================
@st.cache_resource
def load_all():

    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")

    with open("stocks.json") as f:
        STOCKS = json.load(f)

    with open("feature_columns.json") as f:
        FEATURE_COLUMNS = json.load(f)

    with open("config.json") as f:
        CONFIG = json.load(f)

    SEQ_LEN = CONFIG.get("SEQ_LEN", 20)

    # 🔥 BUILD MODEL SAME AS TRAINING
    model = build_model((SEQ_LEN, len(FEATURE_COLUMNS)), len(STOCKS))

    # 🔥 LOAD WEIGHTS
    model.load_weights("model/model.weights.h5")

    return model, scaler_X, scaler_y, STOCKS, FEATURE_COLUMNS, SEQ_LEN


model, scaler_X, scaler_y, STOCKS, FEATURE_COLUMNS, SEQ_LEN = load_all()


# =========================
# 📈 PREDICTION FUNCTION (NOTEBOOK LOGIC)
# =========================
def predict_live():

    df = yf.download(STOCKS, period="60d")["Close"]

    # 🔥 FIX MULTI-INDEX ISSUE
    if isinstance(df, pd.Series):
        df = df.to_frame()

    df = df.dropna()

    df_feat = df.copy()

    for s in STOCKS:
        df_feat[f"{s}_RET"] = df_feat[s].pct_change()
        df_feat[f"{s}_MA7"] = df_feat[s].rolling(7).mean()
        df_feat[f"{s}_MA21"] = df_feat[s].rolling(21).mean()
        df_feat[f"{s}_STD"] = df_feat[s].rolling(21).std()
        df_feat[f"{s}_MOM"] = df_feat[s] - df_feat[s].shift(5)
        df_feat[f"{s}_ROC"] = df_feat[s].pct_change(5)

        # No sentiment (kept same as notebook fallback)
        df_feat[f"{s}_SENT"] = 0

    df_feat = df_feat.dropna()

    # Ensure all columns exist
    for col in FEATURE_COLUMNS:
        if col not in df_feat.columns:
            df_feat[col] = 0

    df_feat = df_feat[FEATURE_COLUMNS]

    X = scaler_X.transform(df_feat)
    X = X[-SEQ_LEN:].reshape(1, SEQ_LEN, X.shape[1])

    pred_reg, pred_cls = model.predict(X, verbose=0)

    pred = scaler_y.inverse_transform(pred_reg)[0]

    # 🔥 SAME POST-PROCESSING (IMPORTANT)
    pred = pred - np.mean(pred)
    pred = 0.7 * pred + 0.3 * np.mean(pred)
    pred = np.clip(pred, -0.08, 0.08)

    if np.std(pred) > 0:
        pred = pred / np.max(np.abs(pred)) * 0.05

    last_price = df.iloc[-1].values
    next_price = last_price * (1 + pred)

    return last_price, pred, pred_cls[0], next_price


# =========================
# 📊 SIGNAL GENERATION
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
# 🌐 UI
# =========================
st.set_page_config(page_title="AI Stock Dashboard", layout="wide")

st.title("📈 Multi-Stock AI Prediction Dashboard")
st.caption("LSTM + Attention | Real-time Market Prediction")

if st.button("🚀 Run Prediction"):

    with st.spinner("Analyzing market..."):

        last_price, pred, cls, next_price = predict_live()
        signals, confidence = generate_signals(pred, cls)

    st.success("Prediction Complete!")

    cols = st.columns(len(STOCKS))

    for i, s in enumerate(STOCKS):

        with cols[i]:
            st.metric(
                label=s,
                value=f"₹{last_price[i]:.2f}",
                delta=f"{pred[i]*100:.2f}%"
            )

            st.write(f"📊 Next Price: ₹{next_price[i]:.2f}")
            st.write(f"📢 Signal: **{signals[i]}**")
            st.write(f"🎯 Confidence: {confidence[i]*100:.1f}%")
