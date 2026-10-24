
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_stress.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "stress_model.pkl"

# Load dataset
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# Remove missing rows
df = df.dropna()

# Separate features and target
X = df.drop(columns=["Stress_Level"])
y = df["Stress_Level"]

# Identify categorical and numerical columns
categorical = X.select_dtypes(include=["object"]).columns.tolist()
numerical = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("numerical", "passthrough", numerical)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100, random_state=42
    ))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train and evaluate
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Model training completed!")
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save trained model
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\nModel saved at:", MODEL_PATH)
