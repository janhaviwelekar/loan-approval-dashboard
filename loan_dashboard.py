# loan_dashboard.py

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from explainerdashboard import ClassifierExplainer, ExplainerDashboard



df = pd.read_csv("train.csv")

# Drop rows with missing values for simplicity (or impute if needed)
df = df.dropna()

# Encode target variable
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Drop Loan_ID (not useful)
df = df.drop(['Loan_ID'], axis=1)

# One-hot encode categorical variables
X = pd.get_dummies(df.drop('Loan_Status', axis=1))
X = X.astype(float)  # Ensures compatibility with ExplainerDashboard
y = df['Loan_Status']


# -------------------------------
# STEP 2: Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# STEP 3: Train a Model
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate (Optional)
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------
# STEP 4: Launch Explainer Dashboard
# -------------------------------
explainer = ClassifierExplainer(model, X_test, y_test,
                                 labels=["Not Approved", "Approved"])

ExplainerDashboard(
    explainer,
    title="Loan Approval Dashboard",
    whatif=True,
    shap_interaction=True
).run(port=10000, host="0.0.0.0")