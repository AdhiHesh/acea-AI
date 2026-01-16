import pandas as pd
import numpy as np
import joblib
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ---------------- CONFIG ----------------
OPENWEATHER_API_KEY = "0fff01f23211490ebad520232ea5bd07"
WEATHER_UNITS = "metric"  # Celsius
# ----------------------------------------

# Load model and preprocessors
try:
    model = joblib.load('random_forest_crop_model.joblib')
    scaler = joblib.load('scaler.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
except FileNotFoundError:
    print("⚠️ Model or preprocessor files not found. Ensure all .joblib files exist.")

app = Flask(__name__)
CORS(app)  # ✅ Enable Cross-Origin Resource Sharing

FEATURE_COLUMNS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']


def get_weather_data(city_name):
    """Fetch real-time weather data from OpenWeatherMap."""
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': OPENWEATHER_API_KEY,
        'units': WEATHER_UNITS
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        return None, f"Weather API error ({response.status_code})."
    
    data = response.json()
    try:
        return {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'rainfall': data.get('rain', {}).get('1h', 0.0)
        }, None
    except KeyError:
        return None, "Weather data unavailable."


@app.route('/')
def home():
    """Serve the main web app."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle crop prediction request from JavaScript frontend."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        # Extract input data
        N = float(data.get('N', 0))
        P = float(data.get('P', 0))
        K = float(data.get('K', 0))
        ph = float(data.get('ph', 7))
        temperature = float(data.get('temperature', 25))
        humidity = float(data.get('humidity', 50))
        rainfall = float(data.get('rainfall', 0))
        city_name = data.get('city_name', 'Unknown')

        # Prepare input for model
        features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                columns=FEATURE_COLUMNS)
        input_scaled = scaler.transform(features)
        prediction_encoded = model.predict(input_scaled)
        recommended_crop = label_encoder.inverse_transform(prediction_encoded)[0]

        # Send JSON response
        return jsonify({
            'recommended_crop': recommended_crop,
            'city': city_name,
            'temperature': temperature,
            'humidity': humidity,
            'rainfall': rainfall
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
