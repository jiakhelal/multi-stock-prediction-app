import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import json
import joblib
import tensorflow as tf
import yfinance as yf
import pandas as pd

# -----------------------
# 🧠 CUSTOM LAYER
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
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_all():
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

    return model, scaler_X, scaler_y, STOCKS, FEATURE_COLUMNS

model, scaler_X, scaler_y, STOCKS, FEATURE_COLUMNS = load_all()

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Stock AI", layout="wide")

st.title("📈 AI Stock Prediction Dashboard")

if st.button("Predict Now 🚀"):

    df = yf.download(STOCKS, period="60d")["Close"].dropna()
    df_feat = df.copy()

    for s in STOCKS:
        df_feat[f"{s}_RET"] = df_feat[s].pct_change()
        df_feat[f"{s}_MA7"] = df_feat[s].rolling(7).mean()
        df_feat[f"{s}_MA21"] = df_feat[s].rolling(21).mean()
        df_feat[f"{s}_STD"] = df_feat[s].rolling(21).std()
        df_feat[f"{s}_MOM"] = df_feat[s] - df_feat[s].shift(5)
        df_feat[f"{s}_ROC"] = df_feat[s].pct_change(5)
        df_feat[f"{s}_SENT"] = 0

    df_feat = df_feat.dropna()
    df_feat = df_feat[FEATURE_COLUMNS]

    X = scaler_X.transform(df_feat)
    X = X[-20:].reshape(1, 20, X.shape[1])

    pred_reg, pred_cls = model.predict(X)
    pred = scaler_y.inverse_transform(pred_reg)[0]

    st.subheader("📊 Predictions")

    for i, s in enumerate(STOCKS):
        st.metric(
            label=s,
            value=f"{pred[i]:.4f}",
            delta=f"{pred[i]*100:.2f}%"
        )