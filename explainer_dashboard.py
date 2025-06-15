# explainer_dashboard.py

import pandas as pd
import joblib
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

# Load data and model
df = pd.read_csv("train.csv")
model = joblib.load("loan_model.pkl")

# Prepare features
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df = df.dropna()
X = df[['Gender', 'Married', 'Education', 'ApplicantIncome', 'LoanAmount']]
y = df['Loan_Status'].map({'Y': 1, 'N': 0})

explainer = ClassifierExplainer(model, X, y, labels=["Not Approved", "Approved"])
ExplainerDashboard(
    explainer,
    title="Loan Approval Dashboard",
    whatif=True,
    shap_interaction=False  
).run(port=8050, host="0.0.0.0")


