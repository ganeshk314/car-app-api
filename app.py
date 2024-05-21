from flask import Flask, request, jsonify
import pickle
from joblib import load
import numpy as np
import requests
import logging
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the model and scaler
model = pickle.load(open('d.pkl', 'rb'))
scaler = load('scalar8.save')

# API configuration
EXCHANGE_RATE_API = os.getenv('EXCHANGE_RATE_API', 'https://api.exchangerate-api.com/v4/latest/EUR')

# Author information
'''
Author K.Ganesh
'''

# Function to get the current Euro to INR exchange rate
def get_euro_to_inr_rate():
    try:
        response = requests.get(EXCHANGE_RATE_API)
        data = response.json()
        return data['rates']['INR']
    except Exception as e:
        logging.error(f"Error fetching exchange rate: {e}")
        return 90.34

@app.route('/')
def home():
    return "Hello world"

@app.route('/predict', methods=['POST'])
def predict():
    # Check if request contains JSON data
    if request.is_json:
        data = request.get_json(force=True)
    else:
        # If not JSON, assume form-data
        data = request.form

    # Extracting features from data
    try:
        features = [
            data['abtest'],
            data['vechileType'],
            int(data['Yor']),
            data['gearbox'],
            int(data['PowerPs']),
            data['model'],
            int(data['kilometer']),
            int(data['monthOfRegistration']),
            data['fuelType'],
            data['brand'],
            data['notRepairedDamage']
        ]
    except KeyError as e:
        return jsonify({'error': f'Missing feature: {e}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid value: {e}'}), 400

    # Preprocessing features
    features = scaler.transform([features])

    # Making prediction
    prediction = model.predict(features)

    # Get the current Euro to INR exchange rate
    euro_to_inr = get_euro_to_inr_rate()

    # Convert prediction from euros to Indian rupees
    prediction_in_inr = prediction[0] * euro_to_inr

    # Log prediction
    logging.info(f"Prediction request: {data}")
    logging.info(f"Prediction: {prediction_in_inr} INR")

    # Returning prediction as JSON response
    return jsonify({'The expected resale value in INR is': round(prediction_in_inr, 2)})

if __name__ == '__main__':
    # Bind to PORT if defined, otherwise default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
