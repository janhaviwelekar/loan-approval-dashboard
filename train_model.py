# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

# Load and clean data
df = pd.read_csv("train.csv")
df = df.dropna()
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
df = df.drop(['Loan_ID'], axis=1)

# Prepare features and target
X = pd.get_dummies(df.drop('Loan_Status', axis=1))
X = X.astype(float)
y = df['Loan_Status']

# Save model column structure for prediction alignment
joblib.dump(X.columns.tolist(), "model_columns.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "loan_model.pkl")

# Evaluation
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# SHAP dashboard
explainer = ClassifierExplainer(model, X_test, y_test, labels=["Not Approved", "Approved"])
ExplainerDashboard(explainer, title="Loan Approval Dashboard", whatif=True, shap_interaction=True).run(port=8050, host="0.0.0.0")




