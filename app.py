import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# 🔹 Config
WEATHER_API_KEY = "18a9e977d32e4a7a8e961308252106"
LOCATION = "Pune"
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "oven_time_predictor.pkl")
SENSOR_TARGETS = {
    'WU311': 160, 'WU312': 190, 'WU314': 190,
    'WU321': 190, 'WU322': 190, 'WU323': 190
}
BUFFER_MINUTES = 10

# 🔹 Step 1: Train Model from Real Data
def train_model():
    data = pd.DataFrame([
        {"sensor": "WU311", "start_temp": 32.5, "ambient_temp": 22, "humidity": 86, "target_temp": 160, "time_to_target": 30.0},
        {"sensor": "WU312", "start_temp": 32.56, "ambient_temp": 22, "humidity": 86, "target_temp": 190, "time_to_target": 47.5},
        {"sensor": "WU314", "start_temp": 37.04, "ambient_temp": 22, "humidity": 86, "target_temp": 190, "time_to_target": 149.0},
        {"sensor": "WU321", "start_temp": 36.98, "ambient_temp": 22, "humidity": 86, "target_temp": 190, "time_to_target": 146.0},
        {"sensor": "WU322", "start_temp": 35.14, "ambient_temp": 22, "humidity": 86, "target_temp": 190, "time_to_target": 64.0},
        {"sensor": "WU323", "start_temp": 33.37, "ambient_temp": 22, "humidity": 86, "target_temp": 190, "time_to_target": 86.5},
        {"sensor": "WU311", "start_temp": 33.26, "ambient_temp": 22, "humidity": 66, "target_temp": 160, "time_to_target": 29.5},
        {"sensor": "WU312", "start_temp": 32.63, "ambient_temp": 22, "humidity": 66, "target_temp": 190, "time_to_target": 47.0},
        {"sensor": "WU314", "start_temp": 36.10, "ambient_temp": 22, "humidity": 66, "target_temp": 190, "time_to_target": 84.0},
    ])
    X = data.drop(columns=["time_to_target"])
    y = data["time_to_target"]

    numeric_features = ["start_temp", "ambient_temp", "humidity", "target_temp"]
    categorical_features = ["sensor"]

    preprocessor = ColumnTransformer([
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(), categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    model.fit(X, y)
    features = numeric_features + list(model.named_steps["preprocessor"].transformers_[1][1].get_feature_names_out(["sensor"]))
    joblib.dump((model, features), MODEL_PATH)
    return model, features

# 🔹 Step 2: Weather API with Fallback
def get_weather():
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={LOCATION}"
        response = requests.get(url, timeout=5)
        data = response.json()
        if "error" in data:
            raise Exception(data["error"]["message"])
        return {
            "temp": data["current"]["temp_c"],
            "humidity": data["current"]["humidity"]
        }
    except Exception as e:
        st.warning(f"⚠️ Weather API failed: {str(e)}. Please enter manually.")
        ambient_temp = st.number_input("Enter ambient temperature (°C)", value=22.0)
        humidity = st.number_input("Enter humidity (%)", value=60)
        return {"temp": ambient_temp, "humidity": humidity}

# 🔹 Step 3: Prediction Logic
def predict(sensor, current_temp, weather, model, features):
    target_temp = SENSOR_TARGETS[sensor]
    if current_temp >= target_temp:
        return 0.0
    sensor_columns = [f"sensor_{s}" for s in SENSOR_TARGETS.keys()]
    input_data = pd.DataFrame({
        'start_temp': [current_temp],
        'ambient_temp': [weather['temp']],
        'humidity': [weather['humidity']],
        'target_temp': [target_temp],
        **{col: [0] for col in sensor_columns}
    })
    input_data[f'sensor_{sensor}'] = 1
    return model.predict(input_data[features])[0]

# 🔹 Step 4: Streamlit UI
st.set_page_config(layout="wide")
st.title("🔥 Industrial Oven Heat-Up Predictor")

current_temp = st.number_input("Enter current oven temperature (°C)", min_value=0.0, step=1.0)
target_time_str = st.text_input("Enter desired oven ready time (HH:MM)", value="07:30")
sensor_options = ["All"] + list(SENSOR_TARGETS.keys())
selected_sensor = st.selectbox("Select sensor", sensor_options)

if st.button("Train & Predict"):
    try:
        model, features = train_model()
        st.success(f"✅ Model trained and saved at: {MODEL_PATH}")
        weather = get_weather()
        target_time = datetime.strptime(target_time_str, "%H:%M")

        def show_prediction(sensor):
            result = predict(sensor, current_temp, weather, model, features)
            total_time = result + BUFFER_MINUTES
            startup_time = target_time - timedelta(minutes=total_time)
            st.markdown(
                f"🔹 **{sensor}**: `{total_time:.1f} min` → Start at `{startup_time.strftime('%H:%M')}`"
            )

        if selected_sensor == "All":
            st.subheader("Predictions for All Sensors")
            for sensor in SENSOR_TARGETS:
                show_prediction(sensor)
        else:
            st.subheader(f"Prediction for {selected_sensor}")
            show_prediction(selected_sensor)

    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
