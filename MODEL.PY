import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# 🔹 Step 1: Real-world training data
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

# 🔹 Step 2: Features and target
X = data.drop(columns=["time_to_target"])
y = data["time_to_target"]

# 🔹 Step 3: Preprocessing
numeric_features = ["start_temp", "ambient_temp", "humidity", "target_temp"]
categorical_features = ["sensor"]

preprocessor = ColumnTransformer([
    ("num", "passthrough", numeric_features),
    ("cat", OneHotEncoder(), categorical_features)
])

# 🔹 Step 4: Pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# 🔹 Step 5: Train and save
model.fit(X, y)
features = numeric_features + list(model.named_steps["preprocessor"].transformers_[1][1].get_feature_names_out(["sensor"]))
joblib.dump((model, features), "oven_time_predictor.pkl")

print("✅ Model trained and saved as oven_time_predictor.pkl")
