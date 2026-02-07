import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import load_and_preprocess
X_train, X_test, y_train, y_test = load_and_preprocess()

feedback_path = "feedback/feedback_log.csv"

if not pd.read_csv(feedback_path).empty:
    feedback_df = pd.read_csv(feedback_path)

    X_feedback = feedback_df.drop(columns=["label"])
    y_feedback = feedback_df["label"]


    X_combined = pd.concat(
        [pd.DataFrame(X_train), X_feedback],
        ignore_index=True
    )
    y_combined = pd.concat(
        [pd.Series(y_train), y_feedback],
        ignore_index=True
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_combined, y_combined)

    joblib.dump(model, "models/fraud_model.pkl")
    print("✅ Model retrained using feedback data")

else:
    print("⚠️ No feedback data available for retraining")
