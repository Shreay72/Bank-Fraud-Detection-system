from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

def train_models(X_train, X_test, y_train, y_test):
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, preds)
        if score > best_score:
            best_model = model
            best_score = score
            best_name = name

    print(f"Best model: {best_name} (ROC-AUC = {best_score:.4f})")
    return best_model, best_name

