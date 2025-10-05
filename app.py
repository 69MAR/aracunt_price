from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from flask_cors import CORS
import os
import joblib

# --- Step 1: Initialize Flask App ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- Step 2: Define Model and Data Configuration ---
# This dictionary maps each arecanut type to its corresponding model, scaler, data file,
# and the specific price column to use from the CSV.
CONFIG = {
    'adike': {
        'model_path': 'model_adike.h5',
        'scaler_path': 'scaler_adike.gz',
        'csv_path': 'arecanut.csv', # Corrected path
        'price_column': 'Max Price (Rs./Quintal)' # Adike uses the maximum price
    },
    'patora': {
        'model_path': 'model_patora.h5',
        'scaler_path': 'scaler_patora.gz',
        'csv_path': 'arecanut.csv', # Corrected path
        'price_column': 'Modal Price (Rs./Quintal)' # Patora uses the modal price
    }
}

# Dictionaries to hold the loaded models and scalers
models = {}
scalers = {}

# --- Step 3: Load All Models and Scalers on Startup ---
def load_all_models():
    """Loads all models and scalers defined in the CONFIG."""
    all_loaded = True
    for key, paths in CONFIG.items():
        model_path = paths['model_path']
        scaler_path = paths['scaler_path']

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                print(f"Loading model and scaler for '{key}'...")
                models[key] = load_model(model_path)
                scalers[key] = joblib.load(scaler_path)
                print(f"Successfully loaded '{key}'.")
            except Exception as e:
                print(f"FATAL ERROR: Failed to load model or scaler for '{key}'. Error: {e}")
                all_loaded = False
        else:
            print(f"FATAL ERROR: Files for '{key}' not found. Make sure '{model_path}' and '{scaler_path}' exist.")
            all_loaded = False
    return all_loaded

# --- Step 4: Define Constants ---
SEQUENCE_LENGTH = 30

# --- Step 5: Function to Get Historical Data for a Specific Type ---
def get_last_30_days_data(arecanut_type):
    """Fetches the last 30 days of price data from the relevant CSV for the given type."""
    type_config = CONFIG.get(arecanut_type, {})
    csv_path = type_config.get('csv_path')
    price_column = type_config.get('price_column')

    if not csv_path or not os.path.exists(csv_path):
        return None, f"Data file '{csv_path}' for type '{arecanut_type}' not found."

    if not price_column:
        return None, f"Price column not configured for type '{arecanut_type}'."

    try:
        df = pd.read_csv(csv_path, parse_dates=['Price Date'])
        
        if price_column not in df.columns:
            return None, f"Column '{price_column}' not found in '{csv_path}'."

        df.rename(columns={'Price Date': 'date', price_column: 'price'}, inplace=True)
        df.sort_values('date', inplace=True)
        
        if len(df) < SEQUENCE_LENGTH:
             return None, f"Not enough data for '{arecanut_type}'. Need at least {SEQUENCE_LENGTH} records, but found {len(df)}."

        # Take the last 30 days of prices
        last_30_days = df['price'].values[-SEQUENCE_LENGTH:]
        return last_30_days, None
    except Exception as e:
        return None, f"Error reading data for '{arecanut_type}': {e}"

# --- Step 6: Define the Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a request body is present and is JSON
    if not request.is_json:
        return jsonify({'error': 'Invalid request: payload must be in JSON format.'}), 400

    data = request.get_json()
    arecanut_type = data.get('type') # Safely get the 'type' from the JSON payload

    # Validate the input
    if not arecanut_type:
        return jsonify({'error': "Missing 'type' in request body. Please specify 'adike' or 'patora'."}), 400

    if arecanut_type not in models or arecanut_type not in scalers:
        return jsonify({'error': f"Invalid arecanut type '{arecanut_type}'. Check server logs for loading errors."}), 400

    # Select the correct model and scaler
    model = models[arecanut_type]
    scaler = scalers[arecanut_type]

    # Get the historical data for the specified type
    past_prices, error_message = get_last_30_days_data(arecanut_type)
    if error_message:
        return jsonify({'error': error_message}), 400

    try:
        # 1. Reshape and scale the input data
        past_prices_reshaped = past_prices.reshape(-1, 1)
        scaled_prices = scaler.transform(past_prices_reshaped)

        # 2. Reshape for the LSTM model [samples, timesteps, features]
        input_data = scaled_prices.reshape(1, SEQUENCE_LENGTH, 1)

        # 3. Make a prediction
        predicted_scaled_price = model.predict(input_data)

        # 4. Inverse transform the prediction to get the actual price
        predicted_price = scaler.inverse_transform(predicted_scaled_price)

        return jsonify({
            'type': arecanut_type,
            'predicted_price': round(float(predicted_price[0][0]), 2)
        })

    except Exception as e:
        print(f"Prediction error for type '{arecanut_type}': {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

# --- Step 7: Run the App ---
if __name__ == '__main__':
    # Load all models and scalers before starting the server
    if load_all_models():
        print("All models loaded. Starting Flask server...")
        # Use port 5001 and make it accessible on your network
        app.run(host='0.0.0.0', port=5001, debug=True)
    else:
        print("Server could not start because some models failed to load.")

