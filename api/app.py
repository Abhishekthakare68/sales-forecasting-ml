# api/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="Walmart Sales Forecasting API",
    description="Predict weekly sales using ML model",
    version="1.0.0"
)

# Load model (load once at startup)
model = joblib.load(
    r"C:\Users\abhis\Desktop\sales-forecasting-ml\models\sales_model.pkl"
)

# ---------- Request Schema ----------
class SalesInput(BaseModel):
    Store: int = 0
    Dept: int = 0
    Temperature: float = 0.0
    Fuel_Price: float = 0.0
    CPI: float = 0.0
    Unemployment: float = 0.0
    Size: int = 0
    Year: int = 0
    Month: int = 0
    Day: int = 0
    Week: int = 0


# ---------- Routes ----------
@app.get("/")
def index():
    return "📈 Walmart Sales Forecasting API is running!"


@app.post("/predict")
def predict(data: SalesInput):
    try:
        input_order = [
            'Store', 'Dept', 'Temperature', 'Fuel_Price', 'CPI',
            'Unemployment', 'Size', 'Year', 'Month', 'Day', 'Week'
        ]

        input_data = [getattr(data, col) for col in input_order]

        prediction = model.predict([input_data])[0]

        return {
            "predicted_weekly_sales": round(float(prediction), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
