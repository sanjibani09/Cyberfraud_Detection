from src.data_preprocessing import load_and_preprocess
import joblib
from sklearn.ensemble import RandomForestClassifier

def train_model(progress_callback=None):

    if progress_callback:
        progress_callback(10, "Loading and preprocessing data...")

    X_train, X_test, y_train, y_test, _ = load_and_preprocess()

    if progress_callback:
        progress_callback(40, "Initializing model...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    if progress_callback:
        progress_callback(70, "Training model...")

    model.fit(X_train, y_train)

    if progress_callback:
        progress_callback(90, "Saving trained model...")

    joblib.dump(model, "models/fraud_model.pkl")

    if progress_callback:
        progress_callback(100, "Training completed successfully!")

    return model
