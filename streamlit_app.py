import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="SAR/USD Predictor", layout="wide")
st.title("🇸🇦 Saudi Riyal (SAR) / USD Exchange Rate Predictor")

# --------------------------------------------------
# 1. Load cleaned data with encoding fallback
# --------------------------------------------------
try:
    df = pd.read_csv("SAR_USD_clean.csv", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv("SAR_USD_clean.csv", encoding="ISO-8859-1")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
df = df[(df["Date"] >= "2015-01-01") & (df["Date"] <= "2022-12-31")].reset_index(drop=True)

st.subheader("📄 Dataset Preview")
st.write(df.head())

# --------------------------------------------------
# 2. Load trained model
# --------------------------------------------------
model = joblib.load("sar_usd_rf_model.pkl")

# Features used in the model
features = [
    "lag_1", "lag_7", "lag_30",
    "roll_7", "roll_30",
    "diff_1", "diff_7",
    "std_7", "std_30"
]

# --------------------------------------------------
# 3. Prepare dataset for evaluation
# --------------------------------------------------
df["lag_1"]  = df["SAR=X"].shift(1)
df["lag_7"]  = df["SAR=X"].shift(7)
df["lag_30"] = df["SAR=X"].shift(30)

df["roll_7"]  = df["SAR=X"].rolling(7).mean()
df["roll_30"] = df["SAR=X"].rolling(30).mean()

df["diff_1"] = df["SAR=X"].diff(1)
df["diff_7"] = df["SAR=X"].diff(7)

df["std_7"]  = df["SAR=X"].rolling(7).std()
df["std_30"] = df["SAR=X"].rolling(30).std()

df = df.dropna().reset_index(drop=True)

X = df[features]
y = df["SAR=X"]

split = int(len(df) * 0.8)
X_train, X_val = X.iloc[:split], X.iloc[split:]
y_train, y_val = y.iloc[:split], y.iloc[split:]

# --------------------------------------------------
# 4. Evaluate model
# --------------------------------------------------
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

st.subheader("📊 Model Evaluation")
st.write(f"Mean Squared Error (MSE): {mse:.5f}")
st.write(f"R² Score: {r2:.5f}")

# --------------------------------------------------
# 5. Historical SAR/USD chart
# --------------------------------------------------
st.subheader("📈 Historical SAR/USD Exchange Rate")
st.line_chart(df.set_index("Date")["SAR=X"])

# --------------------------------------------------
# 6. Predict next-day/year SAR/USD
# --------------------------------------------------
st.subheader("🔮 Next-Day Prediction")

last_row = df.iloc[-1]

def generate_next_features(last_row, df):
    lag_1  = last_row["SAR=X"]
    lag_7  = df["SAR=X"].iloc[-7:].mean() if len(df) >= 7 else lag_1
    lag_30 = df["SAR=X"].iloc[-30:].mean() if len(df) >= 30 else lag_1
    roll_7 = df["SAR=X"].iloc[-7:].mean() if len(df) >= 7 else lag_1
    roll_30= df["SAR=X"].iloc[-30:].mean() if len(df) >= 30 else lag_1
    diff_1 = last_row["SAR=X"] - df["SAR=X"].iloc[-2] if len(df) >= 2 else 0
    diff_7 = last_row["SAR=X"] - df["SAR=X"].iloc[-7] if len(df) >= 7 else 0
    std_7  = df["SAR=X"].iloc[-7:].std() if len(df) >= 7 else 0
    std_30 = df["SAR=X"].iloc[-30:].std() if len(df) >= 30 else 0
    
    return pd.DataFrame([[
        lag_1, lag_7, lag_30,
        roll_7, roll_30,
        diff_1, diff_7,
        std_7, std_30
    ]], columns=features)

next_features = generate_next_features(last_row, df)
next_pred = model.predict(next_features)[0]
st.write(f"Predicted SAR/USD for next day/year: **{next_pred:.4f}**")

# --------------------------------------------------
# 7. User input for last-known SAR adjustment
# --------------------------------------------------
st.subheader("✏️ Adjust Last Known SAR Value")
user_sar = st.number_input("Enter last known SAR/USD:", value=float(last_row["SAR=X"]))
last_row_adjusted = last_row.copy()
last_row_adjusted["SAR=X"] = user_sar

next_features_adjusted = generate_next_features(last_row_adjusted, df)
next_pred_adjusted = model.predict(next_features_adjusted)[0]
st.write(f"Predicted SAR/USD with adjusted last value: **{next_pred_adjusted:.4f}**")

# --------------------------------------------------
# 8. Country info
# --------------------------------------------------
st.subheader("🌍 Country Information")
st.write("Country: **Saudi Arabia (SAR)**")
