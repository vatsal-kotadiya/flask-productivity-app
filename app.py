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






# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import datetime

# app = Flask(__name__)
# CORS(app)

# @app.route("/", methods=["GET"])
# def home():
#     return "✅ Flask Productivity API is running!"

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json(force=True)
#         print("📩 Incoming data:", data)

#         app_name = data.get("app_name", "").lower()
#         usage_time = float(data.get("usage_time", 0))
#         time_period = data.get("time_period", "")

#         # Define app groups
#         productive_apps = ["udemy", "notes", "chrome"]
#         distracting_apps = ["instagram", "whatsapp", "snapchat", "youtube"]

#         # Define schedule periods
#         study_hours = ["08:00–12:00", "13:00–17:00", "20:00–22:00"]
#         food_hours = ["07:00–08:00", "12:00–13:00", "19:00–20:00"]
#         exercise_hours = ["06:00–07:00"]
#         sleep_hours = ["22:00–06:00"]

#         # Default
#         result = "Productive"

#         # --- Apply Logic ---
#         # If distracting app during study hours → Distracting
#         if any(a in app_name for a in distracting_apps):
#             if time_period in study_hours:
#                 result = "Distracting"
#             else:
#                 result = "Productive"

#         # If productive app used during food/exercise → Distracting
#         elif any(a in app_name for a in productive_apps):
#             if time_period in food_hours or time_period in exercise_hours:
#                 result = "Distracting"
#             else:
#                 # Overuse check: >20 mins becomes distracting
#                 result = "Productive" if usage_time <= 20 else "Distracting"

#         # Sleep time logic (rest is productive)
#         elif time_period in sleep_hours:
#             result = "Productive"

#         # Everything else → Distracting
#         else:
#             result = "Distracting"

#         # Final clean-up to only return two labels
#         if result.lower() not in ["productive", "distracting"]:
#             result = "Distracting"

#         response = {
#             "app_name": app_name.capitalize(),
#             "usage_time": usage_time,
#             "time_period": time_period,
#             "prediction": result,
#             "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }

#         print("✅ Prediction:", response)
#         return jsonify(response), 200

#     except Exception as e:
#         print("❌ Error:", e)
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)







# Final code 

# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import datetime

app = Flask(__name__)
CORS(app)

# Load model and encoders
model = joblib.load("model.pkl")
app_encoder = joblib.load("app_encoder.pkl")
time_encoder = joblib.load("time_encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")

print("✅ Model and encoders loaded successfully!")

@app.route("/")
def home():
    return "✅ ML Productivity Predictor API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("📩 Incoming data:", data)

        app_name = data.get("app_name", "")
        usage_time = float(data.get("usage_time", 0))
        time_period = data.get("time_period", "")

        # Encode categorical values
        app_encoded = app_encoder.transform([app_name])[0]
        time_encoded = time_encoder.transform([time_period])[0]

        # Prepare dataframe for prediction
        df_in = pd.DataFrame([[app_encoded, usage_time, time_encoded]],
                             columns=["app_name_encoded", "usage_time", "time_period_encoded"])

        # Make prediction
        prediction = model.predict(df_in)[0]
        label = label_encoder.inverse_transform([prediction])[0]

        response = {
            "app_name": app_name,
            "usage_time": usage_time,
            "time_period": time_period,
            "prediction": label,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print("✅ Prediction:", response)
        return jsonify(response), 200

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
