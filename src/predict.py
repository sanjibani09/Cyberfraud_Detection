import joblib
import numpy as np

model = joblib.load("models/fraud_model.pkl")

# Example transaction
sample_transaction = np.random.rand(1, 30)

prediction = model.predict(sample_transaction)

if prediction[0] == 1:
    print("⚠️ Fraudulent Transaction Detected")
else:
    print("✅ Legitimate Transaction")
