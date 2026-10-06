"""
app.py
------
Flask backend. It does 3 things:
  1. Shows the web page          (GET  /)
  2. Predicts diabetes           (POST /predict)
  3. Reports that server is OK   (GET  /health)

Run it with:   python app.py
"""

import json
import os

import numpy as np
from flask import Flask, jsonify, render_template, request

from predictor import Predictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# On Vercel, static files must live in the 'public' folder.
app = Flask(__name__, static_folder="public/static")

# ---------------------------------------------------------------------------
# Load the saved model ONCE when the server starts
# ---------------------------------------------------------------------------
try:
    predictor = Predictor(os.path.join(MODEL_DIR, "deploy_model.json"))
    with open(os.path.join(MODEL_DIR, "model_info.json")) as f:
        model_info = json.load(f)
except FileNotFoundError:
    raise SystemExit(
        "Model files not found in the 'model' folder.\n"
        "Run 'python train_model.py' first to create them."
    )

# The order MUST match the order used during training.
FEATURES = model_info["feature_order"]

# Allowed (min, max) for each field, and whether it may be left empty.
# Optional fields are filled with the training median by the imputer.
RULES = {
    "Pregnancies":              {"min": 0,    "max": 20,   "optional": False, "label": "Pregnancies"},
    "Glucose":                  {"min": 40,   "max": 300,  "optional": False, "label": "Glucose"},
    "BloodPressure":            {"min": 30,   "max": 200,  "optional": False, "label": "Blood Pressure"},
    "SkinThickness":            {"min": 5,    "max": 100,  "optional": True,  "label": "Skin Thickness"},
    "Insulin":                  {"min": 10,   "max": 900,  "optional": True,  "label": "Insulin"},
    "BMI":                      {"min": 10,   "max": 70,   "optional": False, "label": "BMI"},
    "DiabetesPedigreeFunction": {"min": 0.05, "max": 2.5,  "optional": False, "label": "Diabetes Pedigree Function"},
    "Age":                      {"min": 1,    "max": 120,  "optional": False, "label": "Age"},
}


def validate_and_build_row(data):
    """Check the JSON input. Returns (list_of_8_numbers, None) or (None, error_text)."""
    if not isinstance(data, dict):
        return None, "Please send the data as a JSON object."

    row = []
    for name in FEATURES:
        rule = RULES[name]
        value = data.get(name)

        # Empty optional field -> missing value (NaN), filled in later
        if value is None or str(value).strip() == "":
            if rule["optional"]:
                row.append(np.nan)
                continue
            return None, f"{rule['label']} is required."

        # Must be a real number
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None, f"{rule['label']} must be a number."
        if np.isnan(number) or np.isinf(number):
            return None, f"{rule['label']} must be a valid number."

        # In the dataset, 0 in an optional field means "not measured".
        # We treat it as missing so the imputer fills it in.
        if rule["optional"] and number == 0:
            row.append(np.nan)
            continue

        # Must be in a sensible range
        if not (rule["min"] <= number <= rule["max"]):
            return None, f"{rule['label']} must be between {rule['min']} and {rule['max']}."

        row.append(number)

    return row, None


@app.route("/")
def home():
    """Show the main web page."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Simple check that the server is running."""
    return jsonify({"status": "ok", "model": model_info["best_model"]})


@app.route("/predict", methods=["POST"])
def predict():
    """Receive 8 values as JSON and return the prediction as JSON."""
    data = request.get_json(silent=True)  # None if the JSON is broken
    if data is None:
        return jsonify({"success": False, "error": "Invalid request. Please send JSON data."}), 400

    row, error = validate_and_build_row(data)
    if error:
        return jsonify({"success": False, "error": error}), 400

    try:
        probability = float(predictor.predict_proba(row))  # chance of diabetes (0 to 1)
        label = int(probability > 0.5)                      # 1 = diabetes, 0 = no diabetes
    except Exception:
        # Log the real error on the server, but show a friendly message to the user.
        app.logger.exception("Prediction failed")
        return jsonify({"success": False, "error": "Something went wrong while predicting. Please try again."}), 500

    return jsonify({
        "success": True,
        "prediction": label,
        "result": "Diabetes Detected" if label == 1 else "No Diabetes Detected",
        "probability": round(probability * 100, 1),
        "model": model_info["best_model"],
    })


if __name__ == "__main__":
    # debug=True auto-reloads on code changes. Turn it off in production.
    # If port 5000 is busy, change it here (for example to 5001).
    app.run(host="127.0.0.1", port=5000, debug=True)
