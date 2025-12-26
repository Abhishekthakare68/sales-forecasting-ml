# src/train_model.py

import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

TRAIN_PATH = DATA_DIR / "train.csv"
FEATURES_PATH = DATA_DIR / "features.csv"
STORES_PATH = DATA_DIR / "stores.csv"
MODEL_PATH = MODEL_DIR / "sales_model.pkl"

# ---------- Load Data ----------
train = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
features = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
stores = pd.read_csv(STORES_PATH)

# ---------- Merge Data ----------
df = train.merge(features, on=["Store", "Date"], how="left")
df = df.merge(stores, on="Store", how="left")

# ---------- Feature Engineering ----------
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

# ---------- Handle Missing Values ----------
df.dropna(inplace=True)

# ---------- Features & Target ----------
FEATURE_COLUMNS = [
    "Store", "Dept", "Temperature", "Fuel_Price", "CPI",
    "Unemployment", "Size", "Year", "Month", "Day", "Week"
]

X = df[FEATURE_COLUMNS]
y = df["Weekly_Sales"]

# ---------- Train-Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- Train Model ----------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------- Evaluate ----------
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# ---------- Save Model ----------
MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"✅ Model saved at: {MODEL_PATH}")
