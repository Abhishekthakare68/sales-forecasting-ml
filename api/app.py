# api/app.py

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load(r"C:\Users\abhis\Desktop\sales-forecasting-ml\models\sales_model.pkl")

@app.route('/')
def index():
    return "📈 Walmart Sales Forecasting API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        input_order = ['Store', 'Dept', 'Temperature', 'Fuel_Price', 'CPI',
                       'Unemployment', 'Size', 'Year', 'Month', 'Day', 'Week']

        input_data = [data.get(col, 0) for col in input_order]
        prediction = model.predict([input_data])[0]

        return jsonify({"predicted_weekly_sales": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == "__main__":
    app.run(debug=True)
