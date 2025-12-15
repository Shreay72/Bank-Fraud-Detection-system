# ------------------------------------------
# src/data_preprocessing.py
# ------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from src.feature_engineering import preprocess_features

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])]
    df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)
    df['type'] = df['type'].map({'TRANSFER': 0, 'CASH_OUT': 1})

    X = df.drop('isFraud', axis=1)
    y = df['isFraud']

    X = preprocess_features(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)