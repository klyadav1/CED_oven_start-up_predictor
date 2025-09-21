import streamlit as st
import joblib
import pandas as pd
import requests

# Config
MODEL_PATH = "oven_time_predictor.pkl"
WEATHER_API_KEY = "18a9e977d32e4a7a8e961308252106"
LOCATION = "Pune"
SENSOR_TARGETS = {
    'WU311': 160, 'WU312': 190, 'WU314': 190,
    'WU321': 190, 'WU322': 190, 'WU323': 190
}

def get_weather():
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={LOCATION}"
    response = requests.get(url)
    data = response.json()
    return {
        "temp": data["current"]["temp_c"],
        "humidity": data["current"]["humidity"]
    }

def predict(sensor, current_temp, weather, model, features):
    input_data = pd.DataFrame({
        'start_temp': [current_temp],
        'ambient_temp': [weather['temp']],
        'humidity': [weather['humidity']],
        'target_temp': [SENSOR_TARGETS[sensor]],
        'sensor_WU311': [0], 'sensor_WU312': [0], 'sensor_WU314': [0],
        'sensor_WU321': [0], 'sensor_WU322': [0], 'sensor_WU323': [0]
    })
    input_data[f'sensor_{sensor}'] = 1
    return model.predict(input_data[features])[0]

# Streamlit UI
st.title("Industrial Oven Heat-Up Predictor")
current_temp = st.number_input("Enter current oven temperature (deg C)", min_value=0.0, step=1.0)

sensor_options = ["All"] + list(SENSOR_TARGETS.keys())
selected_sensor = st.selectbox("Select sensor", sensor_options)

if st.button("Predict Heat-Up Time"):
    try:
        model, features = joblib.load(MODEL_PATH)
        weather = get_weather()

        if selected_sensor == "All":
            st.subheader("Predictions for All Sensors")
            for sensor in SENSOR_TARGETS:
                try:
                    result = predict(sensor, current_temp, weather, model, features)
                    st.write(f"🔹 Sensor {sensor}: **{result + 10:.1f} minutes**")
                except Exception as e:
                    st.error(f"❌ Error for {sensor}: {e}")
        else:
            result = predict(selected_sensor, current_temp, weather, model, features)
            st.success(f"✅ Predicted time to target for {selected_sensor}: **{result + 10:.1f} minutes**")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
