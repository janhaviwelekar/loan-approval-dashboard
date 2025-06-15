import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load cleaned dataset
df = pd.read_csv("train.csv")

# Preprocessing
X = df.drop(columns=["Loan_ID", "Loan_Status"])
y = df["Loan_Status"].map({"Y": 1, "N": 0})  # encode target

# One-hot encode categorical features
X_encoded = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ✅ Save model
joblib.dump(model, "loan_model.pkl")

# ✅ Save model input column names (very important for Streamlit)
joblib.dump(X_encoded.columns.tolist(), "model_columns.pkl")  # <-- ADD THIS HERE



