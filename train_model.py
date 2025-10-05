# # 📦 Import libraries
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# from sklearn.ensemble import RandomForestClassifier
# import joblib

# # 1️⃣ Load Dataset
# df = pd.read_csv("habit_tracker_dataset.csv")

# # 2️⃣ Encode categorical features with SEPARATE encoders
# task_encoder = LabelEncoder()
# df["Task"] = task_encoder.fit_transform(df["Task"])

# app_encoder = LabelEncoder()
# df["AppUsed"] = app_encoder.fit_transform(df["AppUsed"])

# # 3️⃣ Features & Labels
# X = df[["AppUsed", "Task", "TimeSpent(min)"]]
# y = df["Productive"].map({"Yes": 1, "No": 0})   # Convert Yes/No → 1/0

# # 4️⃣ Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # 5️⃣ Train Random Forest (best performance)
# rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
# rf_model.fit(X_train, y_train)

# # 6️⃣ Evaluate
# y_pred = rf_model.predict(X_test)
# print("\n📊 Random Forest Results:")
# print("Accuracy :", accuracy_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred, average="weighted"))
# print("Recall   :", recall_score(y_test, y_pred, average="weighted"))
# print("F1-score :", f1_score(y_test, y_pred, average="weighted"))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # 7️⃣ Save model + encoders
# joblib.dump(rf_model, "productivity_model.pkl")
# joblib.dump(task_encoder, "task_encoder.pkl")
# joblib.dump(app_encoder, "app_encoder.pkl")

# print("\n✅ Model & encoders saved successfully!")





import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- 1️⃣ Load schedule ---
schedule = pd.read_csv("smart_daily_schedule.csv")

# --- 2️⃣ Define app behavior categories ---
study_apps = ["Udemy", "Notes", "Chrome", "YouTube"]
distracting_apps = ["Instagram", "WhatsApp", "Snapchat"]

rows = []

# --- 3️⃣ Generate smart dataset ---
for _, row in schedule.iterrows():
    time_period = row["Time Range"]
    expected_time = row["Expected Duration (min)"]
    activity = row["Expected Activity"]

    for app in study_apps + distracting_apps:
        # Try different usage patterns
        for usage in [expected_time * 0.5, expected_time, expected_time * 1.5]:
            
            # --- 📊 Rule-based labeling logic ---
            label = "Productive"

            # CASE 1️⃣ — Study time + using social media → Distracting
            if "Study" in activity and app in distracting_apps:
                label = "Distracting"

            # CASE 2️⃣ — Meal/Exercise time + study app → Productive
            elif any(x in activity for x in ["Breakfast", "Lunch", "Dinner", "Exercise"]) and app in study_apps:
                label = "Productive"

            # CASE 3️⃣ — Study time + study app → Productive
            elif "Study" in activity and app in study_apps:
                label = "Productive"

            # CASE 4️⃣ — Study time + social apps + long usage → Distracting
            if app in distracting_apps and usage > expected_time * 0.8:
                label = "Distracting"

            # CASE 5️⃣ — Meal time + long usage of study apps → Distracting
            if any(x in activity for x in ["Breakfast", "Lunch", "Dinner"]) and app in study_apps and usage > expected_time * 1.2:
                label = "Distracting"

            rows.append([app, time_period, usage, label])

# --- 4️⃣ Create DataFrame ---
df = pd.DataFrame(rows, columns=["app_name", "time_period", "usage_time", "label"])

# --- 5️⃣ Encode text features ---
app_encoder = LabelEncoder()
time_encoder = LabelEncoder()

df["app_encoded"] = app_encoder.fit_transform(df["app_name"])
df["time_encoded"] = time_encoder.fit_transform(df["time_period"])

X = df[["app_encoded", "usage_time", "time_encoded"]]
y = df["label"].map({"Productive": 1, "Distracting": 0})

# --- 6️⃣ Train model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 7️⃣ Evaluate ---
y_pred = model.predict(X_test)
print("\n✅ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# --- 8️⃣ Save model and encoders ---
joblib.dump(model, "smart_usage_model.pkl")
joblib.dump(app_encoder, "app_encoder.pkl")
joblib.dump(time_encoder, "time_encoder.pkl")

print("\n✅ Model & Encoders saved successfully!")
