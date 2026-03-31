import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import json
from tensorflow.keras.models import load_model

# =========================
# LOAD FILES
# =========================
MODEL_PATH = "model/final_model.keras"
SCALER_X_PATH = "model/scaler_X.pkl"
SCALER_Y_PATH = "model/scaler_y.pkl"
CONFIG_PATH = "model/config.json"
FEATURE_PATH = "model/feature_columns.json"
STOCKS_PATH = "model/stocks.json"

model = load_model(MODEL_PATH)
scaler_X = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

with open(CONFIG_PATH) as f:
    config = json.load(f)

with open(FEATURE_PATH) as f:
    feature_cols = json.load(f)

with open(STOCKS_PATH) as f:
    stocks = json.load(f)

SEQ_LEN = config["sequence_length"]

# =========================
# FEATURE ENGINEERING (same as notebook)
# =========================
def add_features(df):
    df["MA"] = df["Close"].rolling(10).mean()
    df["STD"] = df["Close"].rolling(10).std()
    df["MOM"] = df["Close"].diff()
    df["ROC"] = df["Close"].pct_change()
    return df

# =========================
# FETCH DATA
# =========================
def fetch_data(ticker):
    df = yf.download(ticker, period="5y", progress=False)
    if df.empty:
        return None
    df = add_features(df)
    df = df.dropna()
    return df

# =========================
# PREPARE INPUT (same pipeline)
# =========================
def prepare_input(df):
    df = df[feature_cols]

    scaled = scaler_X.transform(df)

    X = []
    for i in range(len(scaled) - SEQ_LEN):
        X.append(scaled[i:i+SEQ_LEN])

    X = np.array(X)

    return X, df

# =========================
# PREDICTION
# =========================
def predict_stock(ticker):
    df = fetch_data(ticker)
    if df is None or len(df) < SEQ_LEN:
        return None

    X, df = prepare_input(df)

    pred = model.predict(X, verbose=0)
    pred_price = scaler_y.inverse_transform(pred)

    last_price = df["Close"].iloc[-1]
    next_price = pred_price[-1][0]

    ret = ((next_price - last_price) / last_price) * 100

    return last_price, next_price, ret, df

# =========================
# SIGNAL + INSIGHT
# =========================
def get_signal(ret):
    if ret > 2:
        return "STRONG BUY", "Strong buying momentum detected — price likely to increase"
    elif ret > 0:
        return "BUY", "Mild upward trend — possible growth"
    elif ret > -2:
        return "HOLD", "Market is neutral — no strong trend"
    elif ret > -5:
        return "SELL", "Mild selling pressure — caution advised"
    else:
        return "STRONG SELL", "Strong selling pressure detected — price may decline"

# =========================
# CONFIDENCE (simple but meaningful)
# =========================
def get_confidence(ret):
    return min(abs(ret) * 15, 95)  # scaled confidence

# =========================
# UI
# =========================
st.set_page_config(layout="wide")

st.title("📈 Multi-Stock AI Prediction Dashboard")

st.markdown("### Model Overview")
st.markdown("""
- LSTM + Attention model  
- Multi-stock correlation learning  
- Technical indicators: MA, STD, MOM, ROC  
""")

selected_stock = st.selectbox("Select Stock", stocks)

if st.button("🚀 Run Prediction"):

    results = []

    # Predict all stocks
    for stock in stocks:
        res = predict_stock(stock)
        if res is None:
            continue

        last_price, next_price, ret, df = res
        signal, insight = get_signal(ret)

        results.append({
            "Stock": stock,
            "Return (%)": round(ret, 2),
            "Next Price": round(next_price, 2),
            "Signal": signal
        })

        # Show selected stock details
        if stock == selected_stock:
            st.subheader(f"{stock} Prediction")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Price", f"{last_price:.2f}")
            col2.metric("Return", f"{ret:.2f}%")
            col3.metric("Next", f"{next_price:.2f}")
            col4.metric("Confidence", f"{get_confidence(ret):.1f}%")

            st.success(signal)
            st.info(f"🧠 Model Insight: {insight}")

            st.line_chart(df["Close"])

    # =========================
    # TABLE
    # =========================
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="Return (%)", ascending=False)

    st.subheader("📊 All Stock Predictions")
    st.dataframe(df_results)

    # =========================
    # BEST STOCK
    # =========================
    best = df_results.iloc[0]
    st.success(f"🚀 Best Opportunity: {best['Stock']} ({best['Return (%)']}%)")

    # =========================
    # DOWNLOAD
    # =========================
    csv = df_results.to_csv(index=False)
    st.download_button("📥 Download Predictions", csv, "predictions.csv")

# =========================
# FOOTER
# =========================
st.warning("⚠️ AI prediction — not financial advice")
