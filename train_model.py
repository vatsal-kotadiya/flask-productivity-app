# 📦 Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1️⃣ Load Dataset
df = pd.read_csv("habit_tracker_dataset.csv")

# 2️⃣ Encode categorical features with SEPARATE encoders
task_encoder = LabelEncoder()
df["Task"] = task_encoder.fit_transform(df["Task"])

app_encoder = LabelEncoder()
df["AppUsed"] = app_encoder.fit_transform(df["AppUsed"])

# 3️⃣ Features & Labels
X = df[["AppUsed", "Task", "TimeSpent(min)"]]
y = df["Productive"].map({"Yes": 1, "No": 0})   # Convert Yes/No → 1/0

# 4️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5️⃣ Train Random Forest (best performance)
rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
rf_model.fit(X_train, y_train)

# 6️⃣ Evaluate
y_pred = rf_model.predict(X_test)
print("\n📊 Random Forest Results:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall   :", recall_score(y_test, y_pred, average="weighted"))
print("F1-score :", f1_score(y_test, y_pred, average="weighted"))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7️⃣ Save model + encoders
joblib.dump(rf_model, "productivity_model.pkl")
joblib.dump(task_encoder, "task_encoder.pkl")
joblib.dump(app_encoder, "app_encoder.pkl")

print("\n✅ Model & encoders saved successfully!")
