import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib

# 🔹 Step 1: Prepare your data manually or load from CSV
data = pd.DataFrame([
    # April 7
    {"sensor": "WU311", "start_temp": 35.90, "ambient_temp": 25.3, "humidity": 22, "target_temp": 160, "time_to_target": 32.5},
    {"sensor": "WU312", "start_temp": 37.63, "ambient_temp": 25.3, "humidity": 22, "target_temp": 190, "time_to_target": 47.5},
    {"sensor": "WU314", "start_temp": 43.94, "ambient_temp": 25.3, "humidity": 22, "target_temp": 190, "time_to_target": 112.0},
    # May 5
    {"sensor": "WU311", "start_temp": 33.26, "ambient_temp": 22.0, "humidity": 66, "target_temp": 160, "time_to_target": 29.5},
    {"sensor": "WU312", "start_temp": 32.63, "ambient_temp": 22.0, "humidity": 66, "target_temp": 190, "time_to_target": 47.0},
    {"sensor": "WU314", "start_temp": 36.10, "ambient_temp": 22.0, "humidity": 66, "target_temp": 190, "time_to_target": 84.0},
    # June 2
    {"sensor": "WU311", "start_temp": 32.50, "ambient_temp": 22.0, "humidity": 86, "target_temp": 160, "time_to_target": 30.0},
    {"sensor": "WU312", "start_temp": 32.56, "ambient_temp": 22.0, "humidity": 86, "target_temp": 190, "time_to_target": 47.5},
    {"sensor": "WU314", "start_temp": 37.04, "ambient_temp": 22.0, "humidity": 86, "target_temp": 190, "time_to_target": 149.0},
])

# 🔹 Step 2: Define features and target
X = data.drop(columns=["time_to_target"])
y = data["time_to_target"]

# 🔹 Step 3: Preprocessing
numeric_features = ["start_temp", "ambient_temp", "humidity", "target_temp"]
categorical_features = ["sensor"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(), categorical_features)
    ]
)

# 🔹 Step 4: Build pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# 🔹 Step 5: Train and save
model.fit(X, y)
features = numeric_features + list(model.named_steps["preprocessor"].transformers_[1][1].get_feature_names_out(["sensor"]))
joblib.dump((model, features), "oven_time_predictor.pkl")

print("✅ Model trained and saved as oven_time_predictor.pkl")
