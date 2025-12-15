# ------------------------------------------
# predict.py
# ------------------------------------------
import pandas as pd
from src.feature_engineering import preprocess_features
from src.model_saving import load_model

# Load sample input
data = pd.read_csv("data/sample_input.csv")
processed_data = preprocess_features(data)

# Load model (update model name if needed)
model = load_model("models/RandomForest_model.pkl")

# Predict
predictions = model.predict(processed_data)
print("Predictions:", predictions)