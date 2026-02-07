import shap
import joblib
import pandas as pd

model = joblib.load("models/fraud_model.pkl")

def explain_prediction(sample_df):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_df)

    return explainer, shap_values
