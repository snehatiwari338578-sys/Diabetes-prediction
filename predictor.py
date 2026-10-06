"""
predictor.py
------------
A tiny prediction engine that needs ONLY numpy (no scikit-learn).

Why? Vercel limits how big a deployed app can be. Scikit-learn + SciPy are
too big, so train_model.py exports the trained model into a small JSON file
(model/deploy_model.json), and this file reads that JSON and predicts.

It repeats the same 3 steps the training code used:
  1. fill empty values with the training medians   (imputer)
  2. scale the values                              (scaler)
  3. ask the model                                 (Random Forest or Logistic Regression)
"""

import json

import numpy as np


class Predictor:
    def __init__(self, json_path):
        with open(json_path) as f:
            d = json.load(f)

        self.model_type = d["model_type"]
        self.medians = np.array(d["medians"], dtype=float)  # imputer numbers
        self.mean = np.array(d["scaler_mean"], dtype=float)  # scaler numbers
        self.scale = np.array(d["scaler_scale"], dtype=float)

        if self.model_type == "random_forest":
            # Each tree is stored as plain lists; turn them into numpy arrays.
            self.trees = []
            for t in d["trees"]:
                self.trees.append({
                    "left": np.array(t["left"]),
                    "right": np.array(t["right"]),
                    "feature": np.array(t["feature"]),
                    "threshold": np.array(t["threshold"], dtype=float),
                    "prob": np.array(t["prob"], dtype=float),  # chance of diabetes at each leaf
                })
        else:  # logistic regression
            self.coef = np.array(d["coef"], dtype=float)
            self.intercept = float(d["intercept"])

    def _prepare(self, row):
        """Fill missing values, then scale. Input: 8 numbers (NaN = missing)."""
        x = np.array(row, dtype=float)
        missing = np.isnan(x)
        x[missing] = self.medians[missing]
        return (x - self.mean) / self.scale

    def predict_proba(self, row):
        """Return the model's probability (0 to 1) of diabetes."""
        x = self._prepare(row)

        if self.model_type == "logistic_regression":
            z = float(np.dot(self.coef, x) + self.intercept)
            return 1.0 / (1.0 + np.exp(-z))

        # Random Forest: walk every tree from the top to a leaf, then average.
        x32 = x.astype(np.float32)  # scikit-learn compares trees using float32
        total = 0.0
        for tree in self.trees:
            node = 0
            while tree["left"][node] != -1:  # -1 means "this is a leaf"
                if x32[tree["feature"][node]] <= tree["threshold"][node]:
                    node = tree["left"][node]
                else:
                    node = tree["right"][node]
            total += tree["prob"][node]
        return total / len(self.trees)

    def predict(self, row):
        """Return 1 (diabetes) or 0 (no diabetes)."""
        return int(self.predict_proba(row) > 0.5)
