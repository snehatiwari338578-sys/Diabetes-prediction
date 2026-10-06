# Diabetes Prediction System

A beginner-friendly **Machine Learning web application** that predicts whether a person is likely to have diabetes from 8 health measurements.

**Input -> ML Model -> Prediction -> Result**

> **Disclaimer:** This is an educational ML project and **not a medical diagnosis**. Always consult a qualified doctor.

## Features

- Clean, responsive web interface (HTML, CSS, vanilla JavaScript)
- Prediction without page reload using `fetch()`
- Input validation on both frontend and backend
- Shows the model's probability estimate (not real medical risk)
- Handles impossible zero values in the dataset correctly
- Compares Logistic Regression and Random Forest, saves the better one

## Technologies Used

| Part | Tools |
|------|-------|
| Machine Learning | Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn |
| Backend | Flask, REST API (JSON) |
| Frontend | HTML, CSS, JavaScript |

## Dataset

**Pima Indians Diabetes Dataset** - 768 rows, 9 columns (8 inputs + 1 target `Outcome`). 500 rows are "No Diabetes" (0), 268 are "Diabetes" (1).
The columns Glucose, BloodPressure, SkinThickness, Insulin and BMI contain zeros that are impossible for a living person. They are treated as **missing values** and filled with the median of the training data.

## ML Workflow

1. Load data with Pandas and explore it (shape, missing values, duplicates, statistics)
2. Replace impossible zeros with NaN
3. Split into 80% training / 20% testing (stratified)
4. Fill missing values (median imputer) and scale features (StandardScaler) - learned from training data only
5. Train Logistic Regression and Random Forest (`class_weight="balanced"`)
6. Evaluate: accuracy, precision, recall, F1, confusion matrix
7. Pick the model with the best F1-score (recall breaks ties) and save it with joblib

## Model Results (test set, 154 rows)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Logistic Regression | 0.734 | 0.603 | 0.704 | 0.650 |
| **Random Forest** (selected) | 0.747 | 0.612 | 0.759 | 0.678 |

Results are modest because the dataset is small and the features are limited. That is normal for this dataset.

## Project Structure

```
diabetes-prediction/
├── dataset/diabetes.csv         # Pima Indians dataset
├── model/
│   ├── diabetes_model.pkl       # trained model
│   ├── scaler.pkl               # StandardScaler
│   ├── imputer.pkl              # fills missing values
│   ├── deploy_model.json        # small numpy-only copy of the model used by the web app
│   └── model_info.json          # feature order + model scores
├── notebooks/diabetes_analysis.ipynb
├── reports/                     # charts created by train_model.py
├── public/static/style.css, script.js
├── templates/index.html
├── app.py                       # Flask backend
├── predictor.py                 # numpy-only predictor (keeps the deployed app small)
├── train_model.py               # trains and saves the model
├── test_api.py                  # tests the /predict endpoint
├── requirements.txt             # what the deployed app needs (Flask, NumPy)
├── requirements-dev.txt         # extra libraries for training and the notebook
├── vercel.json                  # Vercel settings
├── GUIDE.md                     # beginner explanations
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/diabetes-prediction.git
cd diabetes-prediction

python -m venv venv
# Windows:      venv\Scripts\activate
# Mac / Linux:  source venv/bin/activate

pip install -r requirements-dev.txt
```

## How to Run

```bash
python train_model.py     # trains the models and saves them in model/ (also exports deploy_model.json)
python app.py             # starts the website
```

Open **http://127.0.0.1:5000** in your browser.

## API

`POST /predict` with JSON:

```json
{"Pregnancies": 2, "Glucose": 150, "BloodPressure": 80, "SkinThickness": 30,
 "Insulin": 120, "BMI": 32, "DiabetesPedigreeFunction": 0.5, "Age": 40}
```

Response:

```json
{"success": true, "prediction": 1, "result": "Diabetes Detected", "probability": 67.4, "model": "Random Forest"}
```

## Deploy on Vercel

1. Push this project to GitHub.
2. On vercel.com choose **Add New -> Project**, import the repository and click **Deploy** (no settings need changing; Vercel detects Flask from `app.py`).
3. Add your live link here: **Live demo:** https://diabetes-prediction-livid.vercel.app/

The deployed app uses only Flask and NumPy (see `predictor.py`), because scikit-learn and SciPy are too large for Vercel. Training still uses scikit-learn locally.

## Screenshots

_Add screenshots of the app here, for example:_ `![Home page](screenshots/home.png)`

## Future Improvements

- Model comparison dashboard
- Feature importance chart on the website
- Prediction history
- Downloadable prediction report
- Deployment on Render or Railway

## Disclaimer

This project is for learning only. It must not be used for medical decisions.
