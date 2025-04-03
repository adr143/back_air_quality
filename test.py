import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from firebase_admin import db

from constant import *

# Load trained LSTM model
model = load_model("air_quality_model_v5.h5", custom_objects={"mse": MeanSquaredError()})

# Load the scaler using Joblib
scaler = joblib.load("air_scaler.pkl")

# Fetch real-time sensor data from Firebase
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

# 🚀 Scale the input using the same scaler used during training
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


print("Predicted gas levels for the next day:", predicted_gas_values)
