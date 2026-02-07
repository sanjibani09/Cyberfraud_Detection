from sklearn.ensemble import IsolationForest
from data_preprocessing import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess()

model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)

model.fit(X_train)

preds = model.predict(X_test)
preds = [1 if x == -1 else 0 for x in preds]

print("✅ Anomaly detection completed successfully")
