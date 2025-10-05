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





# =====================================================
# 📘 Smart Daily Schedule ML Model - Productivity Predictor
# =====================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# ---------------------------------------
# 1️⃣ Load smart_daily_schedule.csv
# ---------------------------------------
df = pd.read_csv("smart_daily_schedule.csv")
print("✅ Schedule loaded successfully:")
print(df.head())

# ---------------------------------------
# 2️⃣ Create training samples
# ---------------------------------------
# We'll simulate multiple user behavior cases
data = []

for _, row in df.iterrows():
    activity = row["Expected Activity"]
    expected = row["Expected Duration (min)"]

    # Generate 100 random samples per activity
    for _ in range(100):
        actual = np.random.randint(5, expected * 2)  # user usage (5 to 2x expected)

        # Labeling logic
        # Too short or too long = Non-Productive
        if expected * 0.7 <= actual <= expected * 1.2:
            label = "Productive"
        else:
            label = "Non-Productive"

        data.append([activity, expected, actual, label])

# Create dataframe
train_df = pd.DataFrame(data, columns=["Activity", "ExpectedTime", "ActualTime", "Label"])

print("\n✅ Training dataset created with", len(train_df), "records")

# ---------------------------------------
# 3️⃣ Encode text data
# ---------------------------------------
le_activity = LabelEncoder()
le_label = LabelEncoder()

train_df["Activity_enc"] = le_activity.fit_transform(train_df["Activity"])
train_df["Label_enc"] = le_label.fit_transform(train_df["Label"])  # 0 = Non-Productive, 1 = Productive

# ---------------------------------------
# 4️⃣ Train-test split
# ---------------------------------------
X = train_df[["Activity_enc", "ExpectedTime", "ActualTime"]]
y = train_df["Label_enc"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------
# 5️⃣ Train RandomForest model
# ---------------------------------------
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------
# 6️⃣ Evaluate model
# ---------------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\n🎯 Model Accuracy: {:.2f}%".format(acc * 100))
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_label.classes_))

# ---------------------------------------
# 7️⃣ Save model + encoder
# ---------------------------------------
with open("schedule_predictor.pkl", "wb") as f:
    pickle.dump((model, le_activity, le_label), f)

print("\n✅ Model saved successfully as 'schedule_predictor.pkl'")

