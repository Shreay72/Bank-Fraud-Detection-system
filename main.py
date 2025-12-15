# ------------------------------------------
# main.py
# ------------------------------------------
from src.data_preprocessing import load_data
from src.model_training import train_models
from src.model_saving import save_model

def main():
    data_path = "data/fraud.csv"
    X_train, X_test, y_train, y_test = load_data(data_path)
    best_model, best_name = train_models(X_train, X_test, y_train, y_test)
    save_model(best_model, f"models/{best_name}_model.pkl")

if __name__ == "__main__":
    main()