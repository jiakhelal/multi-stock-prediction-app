import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import json
from tensorflow.keras.models import load_model

# -------------------- LOAD FILES --------------------
@st.cache_resource
def load_artifacts():
    model = load_model("model/final_model.keras", compile=False)
    scaler_X = joblib.load("model/scaler_X.pkl")
    scaler_y = joblib.load("model/scaler_y.pkl")
    
    with open("model/config.json") as f:
        config = json.load(f)
        
    with open("model/feature_columns.json") as f:
        feature_cols = json.load(f)
        
    with open("model/stocks.json") as f:
        stocks = json.load(f)
        
    return model, scaler_X, scaler_y, config, feature_cols, stocks

model, scaler_X, scaler_y, config, feature_cols, STOCKS = load_artifacts()

SEQ_LEN = config["SEQ_LEN"]

# -------------------- FEATURE ENGINEERING --------------------
def create_features(df):
    df = df.copy()
    
    for col in df.columns:
        df[f"{col}_ret"] = df[col].pct_change()
        df[f"{col}_ma"] = df[col].rolling(5).mean()
        df[f"{col}_std"] = df[col].rolling(5).std()
        df[f"{col}_mom"] = df[col] - df[col].shift(5)
    
    df.dropna(inplace=True)
    return df

# -------------------- PREDICTION --------------------
def predict_all():
    df = yf.download(STOCKS, start="2019-01-01", progress=False)["Close"]
    df.dropna(inplace=True)

    df_feat = create_features(df)

    X_all = df_feat[feature_cols].values
    X_scaled = scaler_X.transform(X_all)

    X_seq = []
    for i in range(SEQ_LEN, len(X_scaled)):
        X_seq.append(X_scaled[i-SEQ_LEN:i])

    X_seq = np.array(X_seq)
    last_seq = X_seq[-1:]

    pred_scaled = model.predict(last_seq, verbose=0)
    pred = scaler_y.inverse_transform(pred_scaled)[0]

    last_prices = df.iloc[-1].values
    returns = (pred - last_prices) / last_prices * 100

    return last_prices, pred, returns, df

# -------------------- UI --------------------
st.set_page_config(page_title="Stock AI Dashboard", layout="wide")

st.title("📈 Multi-Stock AI Prediction Dashboard")

st.markdown("""
### 🧠 Model Overview
- LSTM + Attention model  
- Multi-stock correlation learning  
- Technical indicators: MA, STD, Momentum  
- 📊 Prediction horizon: **Next 1 trading day**
""")

selected_stock = st.selectbox("📊 Select Stock", STOCKS)

if st.button("🚀 Run Prediction"):
    try:
        last_prices, pred, returns, df = predict_all()

        idx = STOCKS.index(selected_stock)

        price = last_prices[idx]
        pred_price = pred[idx]
        ret = returns[idx]

        # SIGNAL LOGIC
        if ret > 2:
            signal = "🟢 STRONG BUY"
        elif ret > 0:
            signal = "🟢 BUY"
        elif ret > -2:
            signal = "🔴 SELL"
        else:
            signal = "🔴 STRONG SELL"

        # HEURISTIC CONFIDENCE (FIXED)
        confidence = min(abs(ret) * 10 + 20, 95)

        st.subheader(f"📊 {selected_stock} Prediction")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("💰 Current Price", f"{price:.2f}")
        col2.metric("📈 Expected Return", f"{ret:.2f}%")
        col3.metric("🔮 Predicted Price", f"{pred_price:.2f}")
        col4.metric("📊 Model Confidence (heuristic)", f"{confidence:.1f}%")

        st.success(signal) if "BUY" in signal else st.error(signal)

        # MODEL INSIGHT
        if ret > 2:
            insight = "Strong upward momentum detected."
        elif ret > 0:
            insight = "Mild upward trend."
        elif ret > -2:
            insight = "Sideways / neutral behavior."
        else:
            insight = "Strong downward pressure."

        st.markdown(f"💡 **Model Insight:** {insight}")

        # ---------------- BEST STOCK ----------------
        best_idx = np.argmax(returns)
        best_stock = STOCKS[best_idx]
        best_return = returns[best_idx]

        st.success(f"🏆 Best Opportunity: {best_stock} ({best_return:.2f}%)")

        # ---------------- TABLE ----------------
        df_out = pd.DataFrame({
            "Stock": STOCKS,
            "Return (%)": returns.round(2),
            "Next Price": pred.round(2)
        })

        def get_signal(r):
            if r > 2: return "STRONG BUY"
            elif r > 0: return "BUY"
            elif r > -2: return "SELL"
            else: return "STRONG SELL"

        df_out["Signal"] = df_out["Return (%)"].apply(get_signal)
        df_out = df_out.sort_values(by="Return (%)", ascending=False)

        st.subheader("📋 All Stock Predictions")
        st.dataframe(df_out, use_container_width=True)

        # ---------------- DOWNLOAD ----------------
        csv = df_out.to_csv(index=False).encode()
        st.download_button("📥 Download Predictions", csv, "predictions.csv")

        # ---------------- CHART ----------------
        st.subheader("📉 Historical Price")
        st.line_chart(df[selected_stock])

        # ---------------- IMPORTANT NOTES ----------------
        st.warning("""
⚠️ **AI Prediction Disclaimer**

- This model is trained on historical stock data and technical indicators  
- Predictions are short-term (next trading day)  
- Confidence score is heuristic, not a probability  
- Market conditions can change rapidly  

👉 This is for educational purposes only — not financial advice
""")

        # ---------------- MODEL PERFORMANCE NOTE ----------------
        st.info("""
📊 **Model Info**

- Trained on multi-stock historical data (since 2019)
- Uses sequence learning (LSTM + Attention)
- Backtesting recommended for real-world validation
""")

    except Exception as e:
        st.error(f"❌ Error: {e}")
