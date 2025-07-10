# src/train_model.py

import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os

# Load and merge datasets
train = pd.read_csv(r"C:\Users\abhis\Desktop\sales-forecasting-ml\data\train.csv", parse_dates=['Date'])
features = pd.read_csv(r"C:\Users\abhis\Desktop\sales-forecasting-ml\data\features.csv", parse_dates=['Date'])
stores = pd.read_csv(r"C:\Users\abhis\Desktop\sales-forecasting-ml\data\stores.csv")

# Merge datasets
df = train.merge(features, on=['Store', 'Date'], how='left')
df = df.merge(stores, on='Store', how='left')

# Feature Engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Week'] = df['Date'].dt.isocalendar().week.astype(int)

# Drop NA rows (optional)
df.dropna(inplace=True)

# Define features and target
features = ['Store', 'Dept', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size', 'Year', 'Month', 'Day', 'Week']
X = df[features]
y = df['Weekly_Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Save model
os.makedirs(r"C:\Users\abhis\Desktop\sales-forecasting-ml\models", exist_ok=True)
joblib.dump(model, r"C:\Users\abhis\Desktop\sales-forecasting-ml\models\sales_model.pkl")
print("✅ Model saved to models/sales_model.pkl")
