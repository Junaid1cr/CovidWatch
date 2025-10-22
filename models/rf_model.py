
import os
import joblib
from sklearn.ensemble import RandomForestClassifier


MODEL_FILE = os.path.join(os.path.dirname(__file__), "rf_model.joblib")


def create_model(random_state=42, n_estimators=100, max_depth=None):
    """
    Create a RandomForestClassifier instance with given hyperparameters
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    return model


def save_model(model, path=MODEL_FILE):
    """
    Save the trained model to disk
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path=MODEL_FILE):
    """
    Load a trained model from disk
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    model = joblib.load(path)
    return model



if __name__ == "__main__":
    
    rf = create_model()
    print("RandomForest model created:", rf)
    
    save_model(rf)
    
    loaded_rf = load_model()
    print("Loaded model:", loaded_rf)
