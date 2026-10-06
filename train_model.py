"""
train_model.py
--------------
Trains two models (Logistic Regression and Random Forest) on the Pima Indians
Diabetes dataset, compares them, and saves the best one with joblib.

Run it from the project folder:   python train_model.py
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")  # lets matplotlib save pictures without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from predictor import Predictor  # our small numpy-only predictor (used by the web app)

# ---------------------------------------------------------------------------
# 0. Paths (so the script works no matter where you run it from)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "diabetes.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. "
        "Put diabetes.csv inside the 'dataset' folder."
    )

df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------------------------
# 2. Explore the dataset
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: EXPLORING THE DATA")
print("=" * 60)
print("Shape (rows, columns):", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values (NaN) per column:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nBasic statistics:\n", df.describe().round(2))
print("\nOutcome counts (0 = No Diabetes, 1 = Diabetes):\n", df["Outcome"].value_counts())

# Zeros that are impossible for a living person are really "missing" values.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print("\nZero values in columns where 0 is impossible:")
print((df[ZERO_AS_MISSING] == 0).sum())

# ---------------------------------------------------------------------------
# 3. Basic charts (saved as pictures in the 'reports' folder)
# ---------------------------------------------------------------------------
plt.figure(figsize=(5, 4))
sns.countplot(x="Outcome", data=df)
plt.title("Class balance (0 = No Diabetes, 1 = Diabetes)")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "class_balance.png"), dpi=120)
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation between columns")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "correlation_heatmap.png"), dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# 4. Separate features (X) and target (y), then train-test split
# ---------------------------------------------------------------------------
# Handle impossible zeros: turn them into NaN so we can fill them properly.
df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)

X = df.drop("Outcome", axis=1)  # the 8 inputs
y = df["Outcome"]               # the answer we want to predict
FEATURE_NAMES = list(X.columns)

# 80% training, 20% testing. stratify=y keeps the diabetes ratio equal in both.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTraining rows:", X_train.shape[0], "| Testing rows:", X_test.shape[0])

# ---------------------------------------------------------------------------
# 5. Preprocessing: fill missing values (median) and scale
# ---------------------------------------------------------------------------
# IMPORTANT: we learn the medians / mean / std ONLY from the training data.
# Then we apply the same numbers to the test data. This avoids "data leakage".
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train.values)  # .values = plain numbers (no column names)
X_test_imp = imputer.transform(X_test.values)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

# ---------------------------------------------------------------------------
# 6. Train the models
# ---------------------------------------------------------------------------
# class_weight="balanced" tells the model that missing a diabetic person is
# worse than a false alarm. It helps recall.
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
    ),
}

results = {}
print("\n" + "=" * 60)
print("STEP 2: TRAINING AND EVALUATING MODELS")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n--- {name} ---")
    for metric, value in results[name].items():
        print(f"{metric:>10}: {value:.3f}")
    print("Confusion matrix [[TN FP] [FN TP]]:\n", cm)
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

    # Save confusion matrix picture
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Diabetes", "Diabetes"],
                yticklabels=["No Diabetes", "Diabetes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    fname = "confusion_matrix_" + name.lower().replace(" ", "_") + ".png"
    plt.savefig(os.path.join(REPORT_DIR, fname), dpi=120)
    plt.close()

# ---------------------------------------------------------------------------
# 7. Pick the better model
# ---------------------------------------------------------------------------
# For a medical problem we care about catching diabetic people (recall), but
# we also don't want too many false alarms (precision). F1-score balances both,
# so we pick the model with the highest F1. Recall breaks any tie.
best_name = max(results, key=lambda n: (results[n]["f1"], results[n]["recall"]))
best_model = models[best_name]
print("\n" + "=" * 60)
print(f"BEST MODEL: {best_name}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 8. Save everything the web app needs
# ---------------------------------------------------------------------------
joblib.dump(best_model, os.path.join(MODEL_DIR, "diabetes_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer.pkl"))

# Also save the feature order and scores (the Flask app shows them).
info = {
    "best_model": best_name,
    "feature_order": FEATURE_NAMES,
    "metrics": {n: {k: round(v, 3) for k, v in r.items()} for n, r in results.items()},
}
with open(os.path.join(MODEL_DIR, "model_info.json"), "w") as f:
    json.dump(info, f, indent=2)

print("Saved: model/diabetes_model.pkl, model/scaler.pkl, model/imputer.pkl, model/model_info.json")

# ---------------------------------------------------------------------------
# 9. Export a small numpy-only copy of the model for the web app / Vercel
# ---------------------------------------------------------------------------
# The web app does not need scikit-learn. We write the model's numbers to JSON.
deploy = {
    "medians": imputer.statistics_.tolist(),
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
}
if best_name == "Random Forest":
    deploy["model_type"] = "random_forest"
    deploy["trees"] = []
    for est in best_model.estimators_:
        t = est.tree_
        counts = t.value[:, 0, :]
        deploy["trees"].append({
            "left": t.children_left.tolist(),
            "right": t.children_right.tolist(),
            "feature": t.feature.tolist(),
            "threshold": t.threshold.tolist(),
            "prob": (counts[:, 1] / counts.sum(axis=1)).tolist(),
        })
else:
    deploy["model_type"] = "logistic_regression"
    deploy["coef"] = best_model.coef_[0].tolist()
    deploy["intercept"] = float(best_model.intercept_[0])

deploy_path = os.path.join(MODEL_DIR, "deploy_model.json")
with open(deploy_path, "w") as f:
    json.dump(deploy, f)

# Safety check: the numpy predictor must give the same answers as scikit-learn.
check = Predictor(deploy_path)
sk_probs = best_model.predict_proba(scaler.transform(imputer.transform(X.values)))[:, 1]
my_probs = np.array([check.predict_proba(r) for r in X.values])
biggest_gap = float(np.abs(sk_probs - my_probs).max())
print(f"Export check: biggest difference from scikit-learn = {biggest_gap:.2e}")
assert biggest_gap < 1e-6, "Exported model does not match scikit-learn!"
print("Saved: model/deploy_model.json")
