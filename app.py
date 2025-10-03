# from flask import Flask, request, jsonify
# import joblib

# app = Flask(__name__)
# model = joblib.load("productivity_model.pkl")  # load your saved model

# # encode category (same as training)
# def encode(cat):
#     mapping = {'social':0, 'work':1, 'games':2}
#     return mapping.get(cat.lower(), 3)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     usage = float(data['usage_time'])
#     sessions = float(data['session_count'])
#     category = data['app_name']
    
#     X = [[usage, sessions, encode(category)]]
#     pred = model.predict(X)   # 0 = Non-Productive, 1 = Productive
#     label = "Productive" if int(pred[0]) == 1 else "Non-Productive"
    
#     return jsonify({"prediction": label})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)


# from flask import Flask, request, jsonify
# import joblib

# app = Flask(__name__)

# # Load your saved model
# model = joblib.load("productivity_model.pkl")  # make sure this file is in the same folder

# # Example category encoding (match how you trained the model)
# def encode(cat):
#     mapping = {'social':0, 'work':1, 'games':2}  # add your categories
#     return mapping.get(cat.lower(), 3)  # default to 3 if unknown

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     usage = float(data['usage_time'])
#     sessions = float(data['session_count'])
#     category = data['app_name']

#     # Prepare features for model
#     X = [[usage, sessions, encode(category)]]
#     pred = model.predict(X)           # 0 = Non-Productive, 1 = Productive
#     label = "Productive" if int(pred[0]) == 1 else "Non-Productive"

#     return jsonify({"prediction": label})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)




############## this code working first 

# from flask import Flask, request, jsonify
# import joblib

# app = Flask(__name__)

# # Load model
# model = joblib.load("productivity_model.pkl")

# # Example category encoding
# def encode(cat):
#     mapping = {'social':0, 'work':1, 'games':2}
#     return mapping.get(cat.lower(), 3)

# # Default route (works in browser)
# @app.route("/", methods=["GET"])
# def home():
#     # Example default input
#     usage = 30
#     sessions = 2
#     category = "Social"

#     X = [[usage, sessions, encode(category)]]
#     pred = model.predict(X)
#     label = "Productive" if int(pred[0]) == 1 else "Non-Productive"

#     return jsonify({
#         "message": "Flask API is running 🚀",
#         "default_input": {"usage_time": usage, "session_count": sessions, "app_name": category},
#         "prediction": label
#     })

# # POST route for real predictions
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     usage = float(data['usage_time'])
#     sessions = float(data['session_count'])
#     category = data['app_name']

#     X = [[usage, sessions, encode(category)]]
#     pred = model.predict(X)
#     label = "Productive" if int(pred[0]) == 1 else "Non-Productive"

#     return jsonify({"prediction": label})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)





# from flask import Flask, request, jsonify
# import joblib

# app = Flask(__name__)

# # Load model + encoders
# model = joblib.load("productivity_model.pkl")
# le_task = joblib.load("label_encoder_task.pkl")
# le_app = joblib.load("label_encoder_app.pkl")

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json(force=True)
#     print("Received data:", data)

#     # Extract input
#     usage_time = data["usage_time"]
#     session_count = data["session_count"]
#     task = data["task"]
#     app_name = data["app_name"]

#     # Encode using saved encoders
#     encoded_task = le_task.transform([task])[0]
#     encoded_app = le_app.transform([app_name])[0]

#     # Create feature vector
#     features = [[encoded_app, encoded_task, usage_time, session_count]]

#     # Predict
#     prediction = model.predict(features)[0]
#     label = "Productive" if prediction == 1 else "Non-Productive"

#     # return jsonify({"prediction": label})
#     return jsonify({"error": "Please provide 'usage_time' and 'app_name'"}), 400


# if __name__ == "__main__":
#     app.run(debug=True)






# from flask import Flask, request, jsonify
# import joblib

# # 🔹 Load trained model & encoders
# model = joblib.load("productivity_model.pkl")
# task_encoder = joblib.load("task_encoder.pkl")
# app_encoder = joblib.load("app_encoder.pkl")

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return "🚀 Productivity Prediction API is running!"

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json()

#         usage_time = data.get("usage_time")
#         task = data.get("task")
#         app_name = data.get("app_name")

#         # Validate inputs
#         if usage_time is None or task is None or app_name is None:
#             return jsonify({"error": "Missing required fields"}), 400

#         # Encode categorical inputs
#         try:
#             task_encoded = task_encoder.transform([task])[0]
#         except:
#             return jsonify({"error": f"Task '{task}' not found in training data"}), 400

#         try:
#             app_encoded = app_encoder.transform([app_name])[0]
#         except:
#             return jsonify({"error": f"App '{app_name}' not found in training data"}), 400

#         # Prepare features
#         features = [[app_encoded, task_encoded, usage_time]]

#         # Predict
#         prediction = model.predict(features)[0]
#         result = "Productive" if prediction == 1 else "Non-Productive"

#         return jsonify({"prediction": result})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# # if __name__ == "__main__":
# #     app.run(debug=True)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)







# app.py
from flask import Flask, request, jsonify
import joblib, os, pandas as pd, numpy as np

app = Flask(__name__)

# load model & feature list (make sure these files exist in repo)
model = joblib.load("productivity_model.pkl")
feature_cols = joblib.load("feature_columns.pkl")  # list of columns used during training

@app.route("/")
def home():
    return "OK - ML API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    # accept keys task/app_name/usage_time (same names your Android app will send)
    usage = data.get("usage_time")
    task = data.get("task")
    app_name = data.get("app_name")

    if usage is None or task is None or app_name is None:
        return jsonify({"error":"Provide usage_time, task, app_name"}), 400

    try:
        usage = float(usage)
    except:
        return jsonify({"error":"usage_time must be numeric"}), 400

    # Build input row like during training (one-hot columns present in feature_cols)
    row = {c: 0 for c in feature_cols}
    # numeric features (example names — adapt if your feature names differ)
    if "usage_time" in row:
        row["usage_time"] = usage
    if "session_count" in row:
        row["session_count"] = data.get("session_count", 1)
    if "time_per_session" in row:
        # if session_count provided, compute; otherwise fallback 1
        sc = float(data.get("session_count", 1))
        row["time_per_session"] = usage / (sc if sc != 0 else 1)

    # set one-hot columns (they were named like "app_name_<NAME>" and "task_<NAME>")
    app_col = f"app_name_{app_name}"
    task_col = f"task_{task}"
    if app_col in row:
        row[app_col] = 1
    if task_col in row:
        row[task_col] = 1

    df_in = pd.DataFrame([row])[feature_cols]  # ensure same column order
    pred = model.predict(df_in)[0]
    label = "Productive" if int(pred) == 1 else "Non-Productive"
    return jsonify({"prediction": label})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
