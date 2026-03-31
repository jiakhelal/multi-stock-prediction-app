import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
import json
import joblib
import tensorflow as tf
import pandas as pd

# =========================
# 🧠 CUSTOM ATTENTION
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
# LOAD FILES
# =========================
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

    with open("model/config.json") as f:
        CONFIG = json.load(f)
        SEQ_LEN = CONFIG.get("SEQ_LEN", 20)

    return model, scaler_X, scaler_y, STOCKS, FEATURE_COLUMNS, SEQ_LEN


model, scaler_X, scaler_y, STOCKS, FEATURE_COLUMNS, SEQ_LEN = load_all()

# =========================
# UI
# =========================
st.set_page_config(page_title="Stock AI", layout="wide")

st.title("📈 Multi-Stock AI Prediction Dashboard")

st.markdown("""
### 🤖 Model Overview
- LSTM + Attention model  
- Multi-stock correlation learning  
- Technical indicators: MA, STD, MOM, ROC  
""")

selected_stock = st.selectbox("📊 Select Stock", STOCKS)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def fetch_data():
    try:
        df = pd.read_csv("stock_data.csv", index_col=0, parse_dates=True)
        df = df.dropna()
        return df
    except:
        return None

# =========================
# EXPLANATION
# =========================
def explain(pred):
    if pred > 0.02:
        return "Strong upward momentum detected across correlated stocks."
    elif pred > 0:
        return "Moderate bullish trend observed."
    elif pred < -0.02:
        return "Strong downward pressure in market."
    else:
        return "Market shows neutral behavior."

# =========================
# PREDICTION
# =========================
def predict():

    df = fetch_data()
    if df is None:
        st.error("❌ Data not available")
        return None

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

    if len(df_feat) < SEQ_LEN:
        st.error("❌ Not enough data")
        return None

    for col in FEATURE_COLUMNS:
        if col not in df_feat.columns:
            df_feat[col] = 0

    df_feat = df_feat[FEATURE_COLUMNS]

    X = scaler_X.transform(df_feat)
    X = X[-SEQ_LEN:].reshape(1, SEQ_LEN, X.shape[1])

    pred_reg, pred_cls = model.predict(X, verbose=0)

    pred = scaler_y.inverse_transform(pred_reg)[0]

    # smoothing
    pred = pred - np.mean(pred)
    pred = 0.7 * pred + 0.3 * np.mean(pred)
    pred = np.clip(pred, -0.08, 0.08)

    if np.std(pred) > 0:
        pred = pred / np.max(np.abs(pred)) * 0.05

    last_price = df.iloc[-1].values
    next_price = last_price * (1 + pred)

    return df, last_price, pred, pred_cls[0], next_price

# =========================
# RUN
# =========================
if st.button("🚀 Run Prediction"):

    with st.spinner("Running AI model..."):
        result = predict()

    if result is None:
        st.stop()

    df, last_price, pred, cls, next_price = result

    idx = STOCKS.index(selected_stock)

    st.subheader(f"📊 {selected_stock} Prediction")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("💰 Price", f"{last_price[idx]:.2f}")
    col2.metric("📈 Return", f"{pred[idx]*100:.2f}%")
    col3.metric("🔮 Next", f"{next_price[idx]:.2f}")

    # improved confidence
    confidence = min(abs(pred[idx]) * 12 + abs(cls[idx] - 0.5), 0.9)
    col4.metric("🎯 Confidence", f"{confidence*100:.1f}%")

    # SIGNAL
    if pred[idx] > 0.02:
        signal = "STRONG BUY"
        st.success(f"📢 {signal}")
    elif pred[idx] > 0.005:
        signal = "BUY"
        st.success(f"📢 {signal}")
    elif pred[idx] < -0.02:
        signal = "STRONG SELL"
        st.error(f"📢 {signal}")
    elif pred[idx] < -0.005:
        signal = "SELL"
        st.error(f"📢 {signal}")
    else:
        signal = "HOLD"
        st.info(f"📢 {signal}")

    # explanation
    st.write("🧠 Model Insight:")
    st.caption(explain(pred[idx]))

    # =========================
    # 🚀 BEST STOCK
    # =========================
    best_idx = np.argmax(pred)
    st.success(f"🚀 Best Opportunity: {STOCKS[best_idx]} ({pred[best_idx]*100:.2f}%)")

    # =========================
    # 📊 TABLE
    # =========================
    def get_signal(p):
        if p > 0.02:
            return "STRONG BUY"
        elif p > 0.005:
            return "BUY"
        elif p < -0.02:
            return "STRONG SELL"
        elif p < -0.005:
            return "SELL"
        else:
            return "HOLD"

    df_result = pd.DataFrame({
        "Stock": STOCKS,
        "Return (%)": np.round(pred * 100, 2),
        "Next Price": np.round(next_price, 2)
    })

    df_result["Signal"] = df_result["Return (%)"].apply(lambda x: get_signal(x/100))

    st.subheader("📊 All Stock Predictions")
    st.dataframe(df_result, use_container_width=True)

    # =========================
    # 📥 DOWNLOAD
    # =========================
    csv = df_result.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Predictions",
        csv,
        "stock_predictions.csv",
        "text/csv"
    )

    # =========================
    # 📉 CHART
    # =========================
    st.line_chart(df[selected_stock])

    st.warning("⚠️ AI prediction — not financial advice")
