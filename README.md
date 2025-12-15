# Bank Fraud Detection System

This is a web-based application focused on detecting fraudulent transactions using Machine Learning. It utilizes an **XGBoost** model to classify transactions as either "Legit" or "Fraudulent" based on input parameters.

## 🚀 Features
- **Project Structure**: Organized codebase with Flask backend and HTML frontend.
- **Machine Learning**: Uses a pre-trained XGBoost model (`models/XGBoost_model.pkl`) for likely high-accuracy predictions.
- **Web Interface**: Simple and intuitive web form for entering transaction details.
- **Real-time Prediction**: Instant feedback on transaction status.

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Shreay72/Bank-Fraud-Detection-system.git
    cd Bank-Fraud-Detection-system
    ```

2.  **Create and activate a virtual environment** (Recommended)
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

1.  **Run the application**
    ```bash
    python app.py
    ```

2.  **Access the web interface**
    Open your browser and navigate to: `http://127.0.0.1:5000/`

3.  **Test the model**
    Enter the required transaction details (Step, Type, Amount, Balances) and click **Predict**.

## 📂 Project Structure
```
├── models/
│   └── XGBoost_model.pkl   # Pre-trained ML model
├── templates/
│   └── index.html          # Frontend HTML file
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 📝 Technologies
- **Python**
- **Flask**
- **XGBoost**
- **Scikit-learn**
- **Pandas**
- **HTML/CSS**
