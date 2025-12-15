from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("models/XGBoost_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        step = int(request.form["step"])
        type_ = int(request.form["type"])
        amount = float(request.form["amount"])
        oldbalanceOrg = float(request.form["oldbalanceOrg"])
        newbalanceOrig = float(request.form["newbalanceOrig"])
        oldbalanceDest = float(request.form["oldbalanceDest"])
        newbalanceDest = float(request.form["newbalanceDest"])

        # Create DataFrame
        input_data = pd.DataFrame([{
            "step": step,
            "type": type_,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
            "errorBalanceOrig": oldbalanceOrg - newbalanceOrig - amount,
            "errorBalanceDest": newbalanceDest - oldbalanceDest - amount
        }])

        # Predict
        prediction = model.predict(input_data)[0]
        result = "Fraudulent Transaction 🚨" if prediction == 1 else "Legit Transaction ✅"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
