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






from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# --- Load Model ---
try:
    model = joblib.load("smart_usage_model.pkl")
    app_encoder = joblib.load("app_encoder.pkl")
    time_encoder = joblib.load("time_encoder.pkl")
    print("✅ Model & encoders loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)

@app.route('/')
def home():
    return "✅ Smart Productivity API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("📩 Incoming data:", data)

        app_name = data['app_name']
        usage_time = float(data['usage_time'])
        time_period = data['time_period']

        app_encoded = app_encoder.transform([app_name])[0]
        time_encoded = time_encoder.transform([time_period])[0]

        prediction = model.predict([[app_encoded, usage_time, time_encoded]])[0]
        result = "Productive" if prediction == 1 else "Distracting"

        return jsonify({
            "app_name": app_name,
            "usage_time": usage_time,
            "time_period": time_period,
            "prediction": result
        })

    except Exception as e:
        print(f"❌ Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
