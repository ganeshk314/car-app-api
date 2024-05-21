from flask import Flask, request, jsonify
import pickle
from joblib import load
import numpy as np
import requests

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('d.pkl', 'rb'))
scaler = load('scalar8.save')

# Author information
'''
Author K.Ganesh
'''


# Function to get the current Euro to INR exchange rate
def get_euro_to_inr_rate():
    try:
        response = requests.get('https://api.exchangerate-api.com/v4/latest/EUR')
        data = response.json()
        return data['rates']['INR']
    except Exception as e:
        print(f"Error fetching exchange rate: {e}")
        # Fallback to a default value if the API call fails
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

    # Preprocessing features
    features = scaler.transform([features])

    # Making prediction
    prediction = model.predict(features)

    # Get the current Euro to INR exchange rate
    euro_to_inr = get_euro_to_inr_rate()

    # Convert prediction from euros to Indian rupees
    prediction_in_inr = prediction[0] * euro_to_inr

    # Returning prediction as JSON response
    return jsonify({'The expected resale value in INR is': round(prediction_in_inr, 2)})


if __name__ == '__main__':
    app.run(debug=True)
