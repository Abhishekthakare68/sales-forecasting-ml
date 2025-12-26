# src/predict.py

import joblib
import pandas as pd
from pathlib import Path

# Absolute / safe model path
MODEL_PATH = Path(
    r"your-original-location"
)

# Load model once
model = joblib.load(MODEL_PATH)

# Feature order MUST match training
FEATURE_ORDER = [
    'Store', 'Dept', 'Temperature', 'Fuel_Price', 'CPI',
    'Unemployment', 'Size', 'Year', 'Month', 'Day', 'Week'
]


def predict_sales(input_dict: dict) -> float:
    """
    Predict weekly sales from input features.

    Args:
        input_dict (dict): feature-value mapping

    Returns:
        float: predicted weekly sales
    """

    # Ensure correct column order
    df = pd.DataFrame([[input_dict.get(col, 0) for col in FEATURE_ORDER]],
                      columns=FEATURE_ORDER)

    prediction = model.predict(df)[0]
    return float(prediction)


# ---------- Standalone Test ----------
if __name__ == "__main__":
    sample_input = {
        'Store': 1,
        'Dept': 1,
        'Temperature': 45.0,
        'Fuel_Price': 3.75,
        'CPI': 211.0,
        'Unemployment': 7.5,
        'Size': 151315,
        'Year': 2012,
        'Month': 12,
        'Day': 7,
        'Week': 49
    }

    result = predict_sales(sample_input)
    print(f"Predicted Weekly Sales: ${result:.2f}")
