from sklearn.ensemble import RandomForestClassifier
import joblib
from data_preprocessing import load_and_preprocess
from train_utils import train_model
X_train, X_test, y_train, y_test,feature_names = load_and_preprocess()

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(feature_names, "models/feature_names.pkl")
print("✅ Model trained and saved successfully!")
