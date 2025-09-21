import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import requests
import joblib
import glob
import os
from datetime import datetime

# Configuration
WEATHER_API_KEY = "18a9e977d32e4a7a8e961308252106"
LOCATION = "Pune"
MODEL_PATH = "oven_time_predictor.pkl"
CSV_PATH = "D:/IndustrialOvenHeatUpPrediction/Research Data CED OVEN/*.CSV"

COLUMNS = [
    'Date', 'Time',
    'PT_ECO.TR01_WU311_B15.AA.R2251_ActValue[°C]',
    'PT_ECO.TR01_WU312_B15.AA.R2251_ActValue[°C]',
    'PT_ECO.TR01_WU314_B15.AA.R2251_ActValue[°C]',
    'PT_ECO.TR01_WU321_B15.AA.R2251_ActValue[°C]',
    'PT_ECO.TR02_WU322_B15.AA.R2251_ActValue[°C]',
    'PT_ECO.TR02_WU323_B15.AA.R2251_ActValue[°C]'
]

SENSOR_TARGETS = {
    'WU311': 160, 'WU312': 190, 'WU314': 190,
    'WU321': 190, 'WU322': 190, 'WU323': 190
}

def get_weather():
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={LOCATION}"
        response = requests.get(url)
        data = response.json()
        return {
            "temp": data["current"]["temp_c"],
            "humidity": data["current"]["humidity"]
        }
    except:
        return {"temp": 25.0, "humidity": 60}

def prepare_training_data(csv_files):
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, delimiter='\t', encoding='utf-16')
            missing_cols = [col for col in COLUMNS if col not in df.columns]
            if missing_cols:
                print(f"⚠️ Missing columns in {file}: {missing_cols}")
                continue
            df['DateTime'] = pd.to_datetime(
                df['Date'] + ' ' + df['Time'],
                format='%d-%b-%y %I:%M:%S %p',
                errors='coerce'
            )
            df = df.dropna(subset=['DateTime'])
            for sensor in SENSOR_TARGETS.keys():
                col = f"PT_ECO.TR01_{sensor}_B15.AA.R2251_ActValue[°C]" if sensor != 'WU323' else f"PT_ECO.TR02_{sensor}_B15.AA.R2251_ActValue[°C]"
                if col in df.columns:
                    target_temp = SENSOR_TARGETS[sensor]
                    try:
                        reach_time = df[df[col] >= target_temp].iloc[0]
                        dfs.append(pd.DataFrame({
                            'sensor': [sensor],
                            'start_temp': [df[col].iloc[0]],
                            'max_temp': [df[col].max()],
                            'time_to_target': [(reach_time['DateTime'] - df['DateTime'].iloc[0]).total_seconds() / 60],
                            'date': [df['DateTime'].iloc[0]]
                        }))
                    except IndexError:
                        continue
        except Exception as e:
            print(f"❌ Error processing {file}: {str(e)}")
            continue
    return pd.concat(dfs) if dfs else pd.DataFrame()

def create_features(df):
    if df.empty:
        return pd.DataFrame()
    features = []
    for _, row in df.iterrows():
        weather = {'temp': 25.0, 'humidity': 60}
        features.append({
            'sensor': row['sensor'],
            'start_temp': row['start_temp'],
            'ambient_temp': weather['temp'],
            'humidity': weather['humidity'],
            'target_temp': SENSOR_TARGETS[row['sensor']],
            'time_to_target': row['time_to_target']
        })
    return pd.DataFrame(features)

def train_model(features):
    features = pd.get_dummies(features, columns=['sensor'])
    feature_names = features.drop('time_to_target', axis=1).columns.tolist()
    X = features.drop('time_to_target', axis=1)
    y = features['time_to_target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"✅ Model trained. MAE: {mean_absolute_error(y_test, preds):.2f} minutes")
    joblib.dump((model, feature_names), MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")
    return model, feature_names

def predict_heating_time(sensor, current_temp):
    model, features = joblib.load(MODEL_PATH)
    weather = get_weather()
    input_data = pd.DataFrame({
        'start_temp': [current_temp],
        'ambient_temp': [weather['temp']],
        'humidity': [weather['humidity']],
        'target_temp': [SENSOR_TARGETS[sensor]],
        'sensor_WU311': [0],
        'sensor_WU312': [0],
        'sensor_WU314': [0],
        'sensor_WU321': [0],
        'sensor_WU322': [0],
        'sensor_WU323': [0]
    })
    input_data[f'sensor_{sensor}'] = 1
    return model.predict(input_data[features])[0]

# Main Execution
if __name__ == "__main__":
    csv_files = glob.glob(CSV_PATH)
    print(f"📁 Found {len(csv_files)} CSV files")
    oven_data = prepare_training_data(csv_files)
    features = create_features(oven_data)
    if not features.empty:
        model, feature_names = train_model(features)
    else:
        raise ValueError("❌ No valid training data could be processed")

    try:
        current_oven_temp = float(input("🌡️ Enter current oven temperature (°C): "))
        sensor_type = input("🔧 Enter sensor (WU311/WU312/WU314/WU321/WU322/WU323): ").strip().upper()
        if sensor_type not in SENSOR_TARGETS:
            raise ValueError("❌ Invalid sensor type")
        prediction = predict_heating_time(sensor=sensor_type, current_temp=current_oven_temp)
        weather = get_weather()
        print(f"\n⏱️ Predicted time to target: {prediction + 10:.1f} minutes")
        print(f"🌦️ Current weather in {LOCATION}: {weather['temp']}°C, {weather['humidity']}% humidity")
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
