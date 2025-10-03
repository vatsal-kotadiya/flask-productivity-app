# import requests

# url = "http://127.0.0.1:5000/predict"

# test_cases = [
#     {"usage_time": 10, "session_count": 1, "app_name": "Work"},     # Should be Productive
#     {"usage_time": 120, "session_count": 10, "app_name": "Social"}, # Likely Non-Productive
#     {"usage_time": 60, "session_count": 4, "app_name": "Games"},    # Non-Productive
#     {"usage_time": 20, "session_count": 2, "app_name": "Work"}      # Productive
# ]

# for case in test_cases:
#     response = requests.post(url, json=case)
#     print(f"Input: {case} → Prediction: {response.json()}")






import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "usage_time": 75, "task": "Meeting", "app_name": "VSCode"
}      

res = requests.post(url, json=data)
print(res.json())
