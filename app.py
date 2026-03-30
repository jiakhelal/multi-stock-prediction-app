import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import json
import joblib
import tensorflow as tf
import yfinance as yf
import pandas as pd

from transformers import pipeline

# -----------------------
# 🧠 CUSTOM ATTENTION
# -----------------------
class Attention(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1))
        self.b = self.add_weight(shape=(input_shape[1], 1))

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)


# -----------------------
# 🧠 BUILD MODEL
# -----------------------
def build_model(input_shape, output_dim):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    x = Attention()(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)

    out_reg = tf.keras.layers.Dense(output_dim, name="reg")(x)
    out_cls = tf.keras.layers.Dense(output_dim, activation="sigmoid", name="cls")(x)

    return tf.keras.Model(inputs, [out_reg, out_cls])


# -----------------------
# 🤖 LOAD FINBERT
# -----------------------
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")


sentiment_model = load_sentiment_model()


def get_sentiment_score(text):
    try:
        result = sentiment_model(text[:512])[0]
        if result["label"] == "positive":
            return result["score"]
        elif result["label"] == "negative":
            return -result["score"]
        else:
            return 0
    except:
        return 0


# -----------------------
# 📂 LOAD EVERYTHING
# -----------------------
@st.cache_resource
def load_all():

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

    input_shape = (SEQ_LEN, len(FEATURE_COLUMNS))
    output_dim = len(STOCKS)

    model = build_model(input_shape, output_dim)

    # ✅ LOAD WEIGHTS
    model.load_weights("model/model.weights.h5")

    return model, scaler_X, scaler_y, STOCKS, FEATURE_COLUMNS, SEQ_LEN


model, scaler_X, scaler_y, STOCKS, FEATURE_COLUMNS, SEQ_LEN = load_all()


# -----------------------
# 📈 PREDICTION
# -----------------------
def predict_live():

    df = yf.download(STOCKS, period="60d")

    # FIX MULTI STOCK
    if len(STOCKS) > 1:
        df = df["Close"]
    else:
        df = df[["Close"]]
        df.columns = STOCKS

    df = df.dropna()
    df_feat = df.copy()

    for s in STOCKS:

        if s not in df_feat.columns:
            continue

        df_feat[f"{s}_RET"] = df_feat[s].pct_change()
        df_feat[f"{s}_MA7"] = df_feat[s].rolling(7).mean()
        df_feat[f"{s}_MA21"] = df_feat[s].rolling(21).mean()
        df_feat[f"{s}_STD"] = df_feat[s].rolling(21).std()
        df_feat[f"{s}_MOM"] = df_feat[s] - df_feat[s].shift(5)
        df_feat[f"{s}_ROC"] = df_feat[s].pct_change(5)

        # 🔥 FINBERT SENTIMENT
        sentiment = get_sentiment_score(f"{s} stock market news")
        df_feat[f"{s}_SENT"] = sentiment

    df_feat = df_feat.dropna()

    # MATCH TRAIN FEATURES
    for col in FEATURE_COLUMNS:
        if col not in df_feat.columns:
            df_feat[col] = 0

    df_feat = df_feat[FEATURE_COLUMNS]

    X = scaler_X.transform(df_feat)
    X = X[-SEQ_LEN:].reshape(1, SEQ_LEN, X.shape[1])

    pred_reg, pred_cls = model.predict(X, verbose=0)
    pred = scaler_y.inverse_transform(pred_reg)[0]

    # SAME NOTEBOOK LOGIC
    pred = pred - np.mean(pred)
    pred = 0.7 * pred + 0.3 * np.mean(pred)
    pred = np.clip(pred, -0.08, 0.08)

    if np.std(pred) > 0:
        pred = pred / np.max(np.abs(pred)) * 0.05

    last_price = df.iloc[-1].values
    next_price = last_price * (1 + pred)

    return last_price, pred, pred_cls[0], next_price


# -----------------------
# 📊 SIGNALS
# -----------------------
def generate_signals(pred, cls):

    signals = []
    confidence = []

    for r, c in zip(pred, cls):
        conf = min(abs(r) * 12 + abs(c - 0.5), 0.9)
        confidence.append(conf)

        if r > 0.02:
            signals.append("🟢 STRONG BUY")
        elif r > 0.005:
            signals.append("🟢 BUY")
        elif r < -0.02:
            signals.append("🔴 STRONG SELL")
        elif r < -0.005:
            signals.append("🔴 SELL")
        else:
            signals.append("🟡 HOLD")

    return signals, confidence


# -----------------------
# 🎨 UI
# -----------------------
st.set_page_config(page_title="AI Stock Dashboard", layout="wide")

st.title("📈 Multi-Stock AI Prediction Dashboard")
st.caption("LSTM + Attention + FinBERT Sentiment")

if st.button("🚀 Run Prediction"):

    with st.spinner("Running AI model..."):

        last_price, pred, cls, next_price = predict_live()
        signals, confidence = generate_signals(pred, cls)

        cols = st.columns(len(STOCKS))

        for i, col in enumerate(cols):
            col.metric(
                STOCKS[i],
                f"₹{round(last_price[i],2)}",
                f"{round(pred[i]*100,2)}%"
            )

        st.markdown("---")

        results = []

        for i, s in enumerate(STOCKS):
            results.append({
                "Stock": s,
                "Price": round(float(last_price[i]), 2),
                "Return %": round(float(pred[i]*100), 2),
                "Next Price": round(float(next_price[i]), 2),
                "Signal": signals[i],
                "Confidence": round(float(confidence[i]), 2)
            })

        df_result = pd.DataFrame(results)

        st.subheader("📊 Detailed Predictions")
        st.dataframe(df_result, use_container_width=True)
