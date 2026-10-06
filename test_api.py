"""
test_api.py - sends sample inputs to the /predict endpoint.
These are SOFTWARE tests only, NOT medical predictions.
Run:  python test_api.py        (Flask server does not need to be running)
"""
from app import app

CASES = {
    "1. Low-value profile":      {"Pregnancies": 1, "Glucose": 85,  "BloodPressure": 66, "SkinThickness": 29, "Insulin": 0,   "BMI": 26.6, "DiabetesPedigreeFunction": 0.351, "Age": 31},
    "2. High-value profile":     {"Pregnancies": 8, "Glucose": 183, "BloodPressure": 64, "SkinThickness": "", "Insulin": "",  "BMI": 33.3, "DiabetesPedigreeFunction": 0.672, "Age": 52},
    "3. Middle profile":         {"Pregnancies": 3, "Glucose": 120, "BloodPressure": 72, "SkinThickness": 25, "Insulin": 100, "BMI": 30.0, "DiabetesPedigreeFunction": 0.4,   "Age": 35},
    "4. Young, low glucose":     {"Pregnancies": 0, "Glucose": 90,  "BloodPressure": 70, "SkinThickness": 20, "Insulin": 80,  "BMI": 22.0, "DiabetesPedigreeFunction": 0.2,   "Age": 22},
    "5. Older, high BMI":        {"Pregnancies": 5, "Glucose": 160, "BloodPressure": 85, "SkinThickness": 35, "Insulin": 200, "BMI": 38.5, "DiabetesPedigreeFunction": 0.9,   "Age": 58},
}
BAD = {
    "Missing Glucose":     {"Pregnancies": 1, "BloodPressure": 66, "BMI": 26.6, "DiabetesPedigreeFunction": 0.3, "Age": 31},
    "Text instead of number": {"Pregnancies": 1, "Glucose": "abc", "BloodPressure": 66, "BMI": 26.6, "DiabetesPedigreeFunction": 0.3, "Age": 31},
    "Out of range BMI":    {"Pregnancies": 1, "Glucose": 100, "BloodPressure": 66, "BMI": 500, "DiabetesPedigreeFunction": 0.3, "Age": 31},
}

client = app.test_client()
print("VALID CASES")
for name, data in CASES.items():
    r = client.post("/predict", json=data)
    print(f"{name:28} -> {r.status_code} {r.get_json()}")
print("\nINVALID CASES (should return 400)")
for name, data in BAD.items():
    r = client.post("/predict", json=data)
    print(f"{name:28} -> {r.status_code} {r.get_json()}")
r = client.post("/predict", data="not json", content_type="application/json")
print(f"{'Broken JSON':28} -> {r.status_code} {r.get_json()}")
