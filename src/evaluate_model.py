import joblib
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import load_and_preprocess
X_train, X_test, y_train, y_test,_ = load_and_preprocess()
model = joblib.load("models/fraud_model.pkl")
y_pred = model.predict(X_test)
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))
print("🧮 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
