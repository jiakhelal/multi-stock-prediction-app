# 📈 Multi-Stock AI Prediction Dashboard

## 📌 About This Project

This is a machine learning project where I built a web app to predict stock prices using deep learning.

The model is trained on multiple stocks together so that it can learn patterns and relationships between them.
It predicts the **next trading step price** and gives a simple buy/sell signal.

---

## 🚀 What This App Does

* Predicts next price of selected stock
* Shows expected return (%)
* Gives AI-based signal (Buy / Sell / Hold)
* Displays stock trend graph
* Compares all stocks together
* Shows best stock opportunity
* Allows downloading prediction results

---

## 🧠 Model Used

* LSTM + Attention model
* Trained on multiple stocks together
* Sequence length: 60

### Features used:

* Close price
* Moving Average (MA)
* Standard Deviation (STD)
* Momentum (MOM)
* Rate of Change (ROC)

---

## ⚙️ Tech Stack

* Python
* TensorFlow / Keras
* Streamlit
* yFinance
* Pandas, NumPy

---

## 📂 Project Structure

```id="c2axhy"
multi-stock-prediction-app/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── stock_data.csv
│
└── model/
    ├── final_model.keras
    ├── scaler_X.pkl
    ├── scaler_y.pkl
    ├── config.json
    ├── feature_columns.json
    └── stocks.json
```

---

## ▶️ How to Run

```bash id="mwdl2i"
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌍 Deployment

The app is deployed using Streamlit Cloud.

---

🔗 Useful Links
👉 Live App:
https://multi-stock-prediction-app-wxrdrr9bndbiwzakfxscir.streamlit.app/

## ⚠️ Note

This project is for learning purposes only.
It should not be used for real financial decisions.



