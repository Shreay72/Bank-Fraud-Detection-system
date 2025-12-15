# ------------------------------------------
# src/feature_engineering.py
# ------------------------------------------
def preprocess_features(df):
    # Feature: difference between old and new balances
    df['errorBalanceOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['errorBalanceDest'] = df['newbalanceDest'] - df['oldbalanceDest']
    return df
