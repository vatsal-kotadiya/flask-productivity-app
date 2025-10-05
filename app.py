# # app.py
# from flask import Flask, request, jsonify
# import joblib, os, pandas as pd

# app = Flask(__name__)

# # Load model & encoders
# model = joblib.load("productivity_model.pkl")
# task_encoder = joblib.load("task_encoder.pkl")
# app_encoder = joblib.load("app_encoder.pkl")

# @app.route("/")
# def home():
#     return "OK - ML API"

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json(force=True)
    
#     # Get input
#     usage = data.get("usage_time")
#     task = data.get("task")
#     app_name = data.get("app_name")

#     if usage is None or task is None or app_name is None:
#         return jsonify({"error": "Provide usage_time, task, app_name"}), 400

#     try:
#         usage = float(usage)
#     except:
#         return jsonify({"error": "usage_time must be numeric"}), 400

#     # Encode categorical features using saved encoders
#     try:
#         task_val = task_encoder.transform([task])[0]
#     except:
#         return jsonify({"error": f"Unknown task: {task}"}), 400

#     try:
#         app_val = app_encoder.transform([app_name])[0]
#     except:
#         return jsonify({"error": f"Unknown app_name: {app_name}"}), 400

#     # Build DataFrame for prediction (same order as trained model)
#     df_in = pd.DataFrame([{
#         "AppUsed": app_val,
#         "Task": task_val,
#         "TimeSpent(min)": usage
#     }])

#     # Predict
#     pred = model.predict(df_in)[0]
#     label = "Productive" if int(pred) == 1 else "Non-Productive"
#     return jsonify({"prediction": label})

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port)






# ============================================================
# 📘 Flask API for Smart Daily Schedule ML Model
# ============================================================

from flask import Flask, request, jsonify
import pickle
import numpy as np

# ---------------------------------------
# 1️⃣ Initialize Flask app
# ---------------------------------------
app = Flask(__name__)

# ---------------------------------------
# 2️⃣ Load trained model and encoders
# ---------------------------------------
try:
    model, le_activity, le_label = pickle.load(open("schedule_predictor.pkl", "rb"))
    print("✅ Model and encoders loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)

# ---------------------------------------
# 3️⃣ Home route
# ---------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Smart Daily Schedule Predictor API is running!"})

# ---------------------------------------
# 4️⃣ Prediction route
# ---------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        activity = data.get("Activity")
        expected_time = float(data.get("ExpectedTime"))
        actual_time = float(data.get("ActualTime"))

        # Check if all inputs are provided
        if not activity or expected_time is None or actual_time is None:
            return jsonify({"error": "Please provide Activity, ExpectedTime, and ActualTime"}), 400

        # Encode activity
        activity_enc = le_activity.transform([activity])[0]

        # Create feature array
        features = np.array([[activity_enc, expected_time, actual_time]])

        # Predict
        pred = model.predict(features)[0]
        label = le_label.inverse_transform([pred])[0]

        # Return result
        result = {
            "Activity": activity,
            "ExpectedTime": expected_time,
            "ActualTime": actual_time,
            "Prediction": label
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------
# 5️⃣ Run the app
# ---------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)