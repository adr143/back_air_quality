import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU mode
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
from flask_mail import Mail, Message

from constant import *
from firebase_admin import db

import joblib
import tensorflow as tf

from tensorflow.keras.models import load_model
import numpy as np

from datetime import datetime, timedelta

import time
import threading

BASE_DIR = os.getcwd()

MODEL_PATH = os.path.join(BASE_DIR, 'air_quality_model_v5.h5')

AQI_MODEL_PATH = os.path.join(BASE_DIR, 'aqi.keras')

model = load_model(MODEL_PATH, compile=False)

aqi_model = load_model(AQI_MODEL_PATH, compile=False)

SCALER_PATH = os.path.join(BASE_DIR, 'air_scaler.pkl')
AQI_SCALER_PATH = os.path.join(BASE_DIR, 'aqi_scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'aqi_laber.pkl')

scaler = joblib.load(SCALER_PATH)
aqi_scaler = joblib.load(AQI_SCALER_PATH)

SEQ_LENGTH = 7  # Number of time steps

app = Flask(__name__)
CORS(app)

# Flask-Mail Configuration (Use your SMTP settings)
app.config["MAIL_SERVER"] = "smtp.gmail.com"  # Change if using another provider
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USE_SSL"] = False
app.config["MAIL_USERNAME"] = "airqualitymonitoring25@gmail.com"  # Replace with your email
app.config["MAIL_PASSWORD"] = "rhlh fpvo mmsl hcmf"  # Use an App Password for Gmail
app.config["MAIL_DEFAULT_SENDER"] = "airqualitymonitoring25@gmail.com"

mail = Mail(app)

COLUMN_MAPPING = {
    "CO": "CO(ppm)",
    "CO2": "CO2(ppm)",
    "NO2": "NO2(ppm)",
    "O3": "O3(ppm)",
    "PM2_5": "PM2.5",
    "PM10": "PM10",
    "SO2": "SO2(ppm)",
    "H2S": "H2S",
    "TVOC": "TVOC(ppb)"
}

def aqi_classification(gas_values):
    sample = [[
        gas_values['CO'], gas_values['CO2'], gas_values['NO2'], gas_values['O3'],
        gas_values['PM2_5'], gas_values['PM10'], gas_values['SO2'],
        gas_values['H2S'], gas_values['TVOC']
    ]]
    
    new_sample_scaled = aqi_scaler.transform(sample)
    print(aqi_scaler.feature_names_in_)

    predictions = aqi_model.predict(new_sample_scaled)

    print(sample)
    print(gas_values)

    return float(predictions[0][0])

def forecast_air_quality():
    data_reference = db.reference(DB_RECORDS)
    sensor_logs = data_reference.get()

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(sensor_logs, orient='index')
    df.index = pd.to_datetime(df.index, format='%Y%m%d%H%M')

    # Sort DataFrame by timestamp
    df.sort_index(inplace=True)

    # Select the last 7 time steps (SEQ_LENGTH = 7)
    if len(df) < 7:
        raise ValueError("Not enough data points. At least 7 readings are required.")

    latest_sequence = df.iloc[-7:].values  # Shape: (7, 9)

    latest_sequence_scaled = scaler.transform(latest_sequence)

    # Reshape for LSTM input: (batch_size, time_steps, features)
    latest_sequence_scaled = np.expand_dims(latest_sequence_scaled, axis=0)  # Shape: (1, 7, 9)

    # 🔮 Predict gas values for the next day
    predicted_values_scaled = model.predict(latest_sequence_scaled)

    # 🔄 Inverse transform to get original scale
    predicted_values = scaler.inverse_transform(predicted_values_scaled)

    # Convert to dictionary format
    gas_parameters = df.columns.tolist()

    predicted_gas_values = {param: max(0, float(value)) for param, value in zip(gas_parameters, predicted_values[0])}
    
    return predicted_gas_values

def get_air_quality_advice(level: float) -> str:
    advice = {
        "Good": "You can go outside and be active. It's a great day!",
        "Moderate": "If you're sensitive to air pollution, consider reducing prolonged or heavy exertion. Everyone else can enjoy outdoor activities as usual.",
        "Unhealthy": (
            "If you have heart/lung conditions, are older, or are a child, reduce prolonged or heavy exertion. Everyone else should take breaks and monitor symptoms.\n"
            "Sensitive groups should avoid heavy exertion and stay indoors if possible. Everyone should reduce outdoor activities."
        ),
        "Dangerous": (
            "Sensitive groups should avoid all outdoor activity. Everyone should limit exertion and stay indoors when possible. Everyone should remain indoors and avoid all outdoor activity. Follow tips to reduce indoor pollution."
        )
    }
    
    return advice.get(level, "Invalid Air Quality Level. Please enter Good, Moderate, Unhealthy, or Dangerous.")

def categorize_aqi(aqi: float) -> str:
    """Categorizes AQI into air quality levels."""
    if aqi <= 50:
        return "Good", "You can go outside and be active. It's a great day!"
    elif aqi <= 100:
        return "Moderate", "If you're sensitive to air pollution, consider reducing prolonged or heavy exertion. Everyone else can enjoy outdoor activities as usual."
    elif aqi <= 200:
        return "Unhealthy", "If you have heart/lung conditions, are older, or are a child, reduce prolonged or heavy exertion. Everyone else should take breaks and monitor symptoms. Sensitive groups should avoid heavy exertion and stay indoors if possible. Everyone should reduce outdoor activities."
    else:
        return "Dangerous", "Sensitive groups should avoid all outdoor activity. Everyone should limit exertion and stay indoors when possible. Everyone should remain indoors and avoid all outdoor activity. Follow tips to reduce indoor pollution."

def send_air_quality_report():
    """Fetch latest air quality data and send emails to subscribers."""
    with app.app_context():
        ref = db.reference(DB_SUBSCRIBER)
        subscribers = ref.get()

        if not subscribers:
            print("No subscribers found.")
            return
        
        print(subscribers)

        emails = [sub.get("email") for sub in subscribers if sub and sub.get("email")]

        if not emails:
            print("No valid email addresses found.")
            return

        # Fetch latest sensor data
        data_reference = db.reference(DB_RECORDS)
        sensor_logs = data_reference.get()

        if not sensor_logs:
            print("No sensor data available.")
            return

        df = pd.DataFrame.from_dict(sensor_logs, orient='index')
        df.index = pd.to_datetime(df.index, format='%Y%m%d%H%M')
        df.sort_index(inplace=True)
        latest_data = df.tail(1).to_dict(orient='records')[0]

        # Prepare email content
        report_body = f"""
        Air Quality Report for {datetime.now().strftime('%B %d, %Y')}:

        CO2: {latest_data.get('CO2')} PPM
        TVOC: {latest_data.get('TVOC')} PPB
        SO2: {latest_data.get('SO2')} PPM
        O3: {latest_data.get('O3')} PPM
        H2S: {latest_data.get('H2S')} PPM
        NO2: {latest_data.get('NO2')} PPM
        CO: {latest_data.get('CO')} PPM
        PM2.5: {latest_data.get('PM2_5')} µg/m³
        PM10: {latest_data.get('PM10')} µg/m³

        Stay safe!
        """

        try:
            msg = Message("Daily Air Quality Report", recipients=emails)
            msg.body = report_body
            mail.send(msg)
            print(f"Email sent to {len(emails)} subscribers.")

        except Exception as e:
            print(f"Error sending email: {e}")

@app.route("/subscribe", methods=["POST"])
def subscribe():
    try:
        # Retrieve JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400

        # Extract fields
        name = data.get("name")
        email = data.get("email")
        number = data.get("number")

        # Reference database path
        ref = db.reference(DB_SUBSCRIBER)

        # Get existing subscribers (ensure it's a dictionary)
        subscribers = ref.get() or {}

        # Generate a new numerical index
        new_index = len(subscribers) + 1

        # Add new subscriber with numerical index
        ref.child(str(new_index)).set({
            "name": name,
            "email": email,
            "number": number
        })

        return jsonify({"message": "Subscription successful!"}), 201

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/number")
def get_number():
    ref = db.reference("tbl_subcribers/subscribers")
    subscribers = ref.get()

    if not subscribers:
        return "No subscribers found", 404

    # Extract numbers while handling missing "number" fields
    numbers = [subscriber.get("number") for subscriber in subscribers]

    if not numbers:
        return "No phone numbers found", 404

    return ",".join(numbers)

@app.route("/air_quality", methods=["GET"])
def air_quality():
    current_date = (datetime.now() + timedelta(days=0)).strftime("%B %d, %Y")
    return render_template("index.html", current_date=current_date)


@app.route("/forecast", methods=["GET"])
def forecast():
    current_date = (datetime.now() + timedelta(days=1)).strftime("%B %d, %Y")
    return render_template("forecast.html", current_date=current_date)


@app.route("/api/sensor-data", defaults={'forecast': False}, methods=["GET"])
@app.route("/api/sensor-data/<forecast>", methods=["GET"])
def get_sensor_data(forecast):
    """API endpoint that returns latest sensor readings."""
    if forecast:
        forecast_result = forecast_air_quality()
        predicted_aqi = aqi_classification(forecast_result)
        aqi_lvl = categorize_aqi(predicted_aqi)
        aqi_class, advice = get_air_quality_advice(aqi_class)
        print(advice)
        sensor_data = [
            { "id": 'CO2_container', "label": 'CO₂ (PPM)', "value": "{0:.2f}".format(forecast_result.get("CO2"))},
            { "id": 'TVOC_container', "label": 'TVOC (PPB)',  "value": "{0:.2f}".format(forecast_result.get("TVOC"))},
            { "id": 'SO2PPM_container', "label": 'SO₂ (PPM)', "value": "{0:.2f}".format(forecast_result.get("SO2"))},
            { "id": 'O3PPM_container', "label": 'O₃ (PPM)', "value": "{0:.2f}".format(forecast_result.get("O3"))},
            { "id": 'H2SPPM_container', "label": 'H₂S (PPM)',  "value": "{0:.2f}".format(forecast_result.get("H2S"))},
            { "id": 'NO2PPM_container', "label": 'NO₂ (PPM)', "value": "{0:.2f}".format(forecast_result.get("NO2"))},
            { "id": 'CO_container', "label": 'CO (PPM)', "value": "{0:.2f}".format(forecast_result.get("CO"))},
            { "id": 'PM2_container', "label": 'PM2.5 (µg/m³)', "value": "{0:.2f}".format(forecast_result.get("PM2_5"))},
            { "id": 'PM10_container', "label": 'PM10 (µg/m³)', "value": "{0:.2f}".format(forecast_result.get("PM10"))},
            { "id": 'aqi_container', "label": 'Air Quality Class', "evaluation": aqi_class, "advice": advice, "aqi": aqi_lvl}
        ]
        return jsonify(sensor_data)
    else:
        data_reference = db.reference(DB_RECORDS)
        sensor_logs = data_reference.get()
        # print(sensor_logs)

        # convert the dictionary to a DataFrame with timestamps as datetime objects
        df = pd.DataFrame.from_dict(sensor_logs, orient='index')
        df.index = pd.to_datetime(df.index, format='%Y%m%d%H%M')

        # sort the DataFrame by the index (default: ascending order)
        df.sort_index(inplace=True)

        latest_data = df.tail(1).to_dict(orient='records')[0]

    # print(latest_data)
    # {'CO': 753, 'CO2': 405, 'H2S': 256, 'NO2': 184, 'O3': 47, 'PM2': 1.09909, 'SO2': 8, 'TVOC': 0}

    # max ppm = 0 - 1000
    #  tvoc 400 µg/m3
    #  pm2.5 35 µg/m3

    max_ppm = 1000

    co2 = int((latest_data.get("CO2") / max_ppm) * 100) if latest_data.get("CO2") != 0 else 0
    so2 = int((latest_data.get("SO2") / max_ppm) * 100) if latest_data.get("SO2") != 0 else 0
    co = int((latest_data.get("CO") / max_ppm) * 100) if latest_data.get("CO") != 0 else 0
    o3 = int((latest_data.get("O3") / max_ppm) * 100) if latest_data.get("O3") != 0 else 0
    h2s = int((latest_data.get("H2S") / max_ppm) * 100) if latest_data.get("H2S") != 0 else 0
    no2 = int((latest_data.get("NO2") / max_ppm) * 100) if latest_data.get("NO2") != 0 else 0

    tvoc = int((latest_data.get("TVOC") / 400) * 100) if latest_data.get("TVOC") != 0 else 0
    pm2 = int((latest_data.get("PM2_5") / 35) * 100) if latest_data.get("PM2_5") != 0 else 0
    pm10 = int((latest_data.get("PM10") / 35) * 100) if latest_data.get("PM10") != 0 else 0

    overall = int((co2 + so2 + co + o3 + h2s + no2 + tvoc + pm2) / 8)

    evaluation = "Good" if overall <= 50 else "Moderate" if overall <= 70 else "Bad" if overall <= 80 else "Unhealthy"
    print(latest_data)
    aqi_level = round(aqi_classification(latest_data), 2)

    aqi_class, advice = categorize_aqi(aqi_level)

    sensor_data = [
            { "id": 'CO2_container', "label": 'CO₂ (PPM)', "percentage": co2, "value": "{0:.2f}".format(latest_data.get("CO2"))},
            { "id": 'TVOC_container', "label": 'TVOC (PPB)', "percentage": tvoc, "value": "{0:.2f}".format(latest_data.get("TVOC"))},
            { "id": 'SO2PPM_container', "label": 'SO₂ (PPM)', "percentage": so2, "value": "{0:.2f}".format(latest_data.get("SO2"))},
            { "id": 'O3PPB_container', "label": 'O₃ (PPB)', "percentage": o3, "value": "{0:.2f}".format(latest_data.get("O3"))},
            { "id": 'H2SPPM_container', "label": 'H₂S (PPM)', "percentage": h2s, "value": "{0:.2f}".format(latest_data.get("H2S"))},
            { "id": 'NO2PPB_container', "label": 'NO₂ (PPB)', "percentage": no2, "value": "{0:.2f}".format(latest_data.get("NO2"))},
            { "id": 'CO_container', "label": 'CO (PPM)', "percentage": co, "value": "{0:.2f}".format(latest_data.get("CO"))},
            { "id": 'PM2_container', "label": 'PM2.5 (µg/m³)', "percentage": pm2, "value": "{0:.2f}".format(latest_data.get("PM2_5"))},
            { "id": 'PM10_container', "label": 'PM10 (µg/m³)', "percentage": pm10, "value": "{0:.2f}".format(latest_data.get("PM10"))},
            { "id": 'overall_container', "label": 'Air Quality', "percentage": overall, "evaluation": evaluation},
            { "id": 'aqi_container', "label": 'Air Quality Class', "percentage": overall, "evaluation": aqi_class, "advice": advice, "aqi": aqi_level}
        ]
    return jsonify(sensor_data)

def schedule_email():
    """Runs a background thread to send the email daily at 8:00 PM."""
    while True:
        now = datetime.now()
        target_time = now.replace(hour=11, minute=46 , second=0, microsecond=0)

        if now > target_time:
            target_time += timedelta(days=1)  # Move to next day if already past 8:00 PM

        wait_time = (target_time - now).total_seconds()
        print(f"Next email scheduled in {wait_time / 3600:.2f} hours.")

        time.sleep(wait_time)  # Sleep until 8:00 PM
        send_air_quality_report()

threading.Thread(target=schedule_email, daemon=True).start()

@app.route("/send-test-email")
def send_test_email():
    """Manually trigger an email for testing."""
    send_air_quality_report()
    return "Test email sent!"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
