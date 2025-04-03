import numpy as np
import os
import joblib
import tensorflow as tf

from tensorflow.keras.models import load_model

BASE_DIR = os.getcwd()

MODEL_PATH = os.path.join(BASE_DIR, 'air_quality_model_v5.h5')

AQI_MODEL_PATH = os.path.join(BASE_DIR, 'aq_class.h5')

model = load_model(MODEL_PATH, compile=False)

aqi_model = load_model(AQI_MODEL_PATH, compile=False)

SCALER_PATH = os.path.join(BASE_DIR, 'air_scaler.pkl')
AQI_SCALER_PATH = os.path.join(BASE_DIR, 'aqi_scaler.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')

scaler = joblib.load(SCALER_PATH)
aqi_scaler = joblib.load(AQI_SCALER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# Example new data (replace with actual values)
new_sample = np.array([[1.5, 400, 2, 50, 30, 10, 17, 15, 300]])  # Example input

# Normalize new data (use the same scaler from training)
new_sample_scaled = aqi_scaler.transform(new_sample)

# Predict class probabilities
predictions = aqi_model.predict(new_sample_scaled)

# Get the predicted class
predicted_class = np.argmax(predictions)  # Get the index of highest probability
predicted_label = label_encoder.inverse_transform([predicted_class])  # Convert index to label

print(predicted_label)
