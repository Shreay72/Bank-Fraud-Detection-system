from flask import Flask, request, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load trained XGBoost model
model_path = os.path.join("models", "XGBoost_model.pkl")
model = joblib.load(model_path)

# Transaction type mapping (must match the one used during training)
type_mapping = {
    "CASH_OUT": 0,
    "PAYMENT": 1,
    "CASH_IN": 2,
    "TRANSFER": 3,
    "DEBIT": 4
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        try:
            # Get form inputs
            type_ = request.form["type"]
            amount = float(request.form["amount"])
            oldbalanceOrg = float(request.form["oldbalanceOrg"])
            newbalanceOrig = float(request.form["newbalanceOrig"])
            oldbalanceDest = float(request.form["oldbalanceDest"])
            newbalanceDest = float(request.form["newbalanceDest"])
            isFlaggedFraud = int(request.form["isFlaggedFraud"])

            # Encode 'type'
            type_encoded = type_mapping.get(type_, -1)

            # Derived features (same as training)
            errorBalanceOrig = oldbalanceOrg - newbalanceOrig
            errorBalanceDest = newbalanceDest - oldbalanceDest

            # Prepare input
            input_data = np.array([[0, type_encoded, amount,
                                    oldbalanceOrg, newbalanceOrig,
                                    oldbalanceDest, newbalanceDest,
                                    errorBalanceOrig, errorBalanceDest]])

            # Make prediction
            pred = model.predict(input_data)[0]
            prediction = "✅ Legitimate Transaction" if pred == 0 else "❌ Fraudulent Transaction"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
