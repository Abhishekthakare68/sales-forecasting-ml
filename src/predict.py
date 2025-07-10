# src/predict.py

import joblib
import pandas as pd

# Load model
model = joblib.load(r"C:\Users\abhis\Desktop\sales-forecasting-ml\models\sales_model.pkl")

def predict_sales(input_dict):
    df = pd.DataFrame([input_dict])
    return model.predict(df)[0]

# Example usage
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

    prediction = predict_sales(sample_input)
    print(f"Predicted Weekly Sales: ${prediction:.2f}")
