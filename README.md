🛡️ Cyber Fraud Detection System using Machine Learning

📌 Project Overview

The Cyber Fraud Detection System is a machine learning–based application designed to detect fraudulent online transactions by analyzing transaction and identity data.
The system predicts the fraud risk score of a transaction, provides decision support (Approved / Review / Block), and explains predictions using Explainable AI (SHAP).

A Streamlit web application is used to make the system interactive and user-friendly.

🎯 Objectives

Detect fraudulent transactions using machine learning

Provide risk-based scoring instead of only binary output

Explain model predictions using SHAP

Visualize fraud risk distribution

Simulate real-world fraud monitoring systems

Key Features

✅ Fraud Risk Prediction

Predicts fraud probability (0–100%)

Uses a trained Random Forest Classifier

✅ Decision Engine

Based on risk score:

10% → ✅ Approved

10–40% → ⚠️ Review Required

40% → ⛔ Block Transaction

✅ Explainable AI (SHAP)

Displays top contributing features for each transaction

Helps understand why a transaction was flagged

✅ Fraud Risk Distribution

Displays a histogram of fraud risk across random transactions

Highlights the selected transaction on the graph

✅ Model Training Interface

Train / Retrain model directly from the web app

Progress bar and training status shown

🧠 Dataset Used

IEEE-CIS Fraud Detection Dataset

Source: Kaggle

Contains:

train_transaction.csv

train_identity.csv

Highly imbalanced real-world fraud dataset

Includes anonymized features (V1–V339) to preserve privacy

Project Structure

<pre>
Cyberfraud_Detection/
│
├── app.py                         # Streamlit web application
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
│
├── data/
│   └── raw/
│       ├── train_transaction.csv
│       ├── train_identity.csv
│
├── models/
│   ├── fraud_model.pkl            # Trained ML model
│   └── feature_names.pkl          # Feature list used in training
│
├── src/
│   ├── data_preprocessing.py      # Data loading & preprocessing
│   ├── train_model.py             # Model training script
│   ├── train_utils.py             # Training utilities
│   ├── evaluate_model.py          # Model evaluation
│   └── shap_utils.py              # SHAP explainability functions
│
└── feedback/
    └── feedback_log.csv            # User feedback 
</pre>

⚙️ Technologies Used

Python 3

Pandas & NumPy – Data processing

Scikit-learn – Machine learning

SHAP – Explainable AI

Matplotlib – Visualization

Streamlit – Web application

Joblib – Model persistence
📦 Installation & Setup

1️⃣ Clone the repository
git clone https://github.com/your-username/Cyberfraud_Detection.git
cd Cyberfraud_Detection

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Train the model
python src/train_model.py

4️⃣ Run the web application
streamlit run app.py

🖥️ How the Application Works

User selects a transaction index

System preprocesses the transaction

Model predicts fraud risk score

Decision is displayed (Approved / Review / Block)

User can:

View explanation (SHAP)

View fraud risk distribution

📊 Model Performance
Accuracy: 93%
Fraud Recall: 70%


High recall ensures that most fraudulent transactions are detected, which is critical in real-world fraud systems.

🧪 Explainability (SHAP)

Shows top features influencing a prediction

Supports transparency and trust

Handles anonymized features safely

Example:
<pre> 
Top contributing features:
• V98
• V99
• V218
• V219
• V178
</pre>

🧠 Real-World Relevance

This system mimics how banks and fintech companies:

Use risk-based scoring

Combine ML + rule-based logic

Provide explainable predictions

Monitor fraud distributions continuously

⚠️ Limitations

Dataset features are anonymized

Real-time streaming data not implemented

Feedback-based retraining is simulated

🔮 Future Enhancements

Risk percentile calculation

Natural language explanation of predictions

Live transaction streaming

Deep learning models

Database integration

Role-based dashboards (Admin / Analyst)
