# Beginner Guide

## How to run (step by step)

1. Install Python 3.9+ from python.org (tick "Add Python to PATH" on Windows).
2. Open the `diabetes-prediction` folder in VS Code (File -> Open Folder).
3. Open the terminal in VS Code (Terminal -> New Terminal).
4. Create a virtual environment: `python -m venv venv`
5. Activate it. Windows: `venv\Scripts\activate`. Mac/Linux: `source venv/bin/activate`. You will see `(venv)` in the terminal.
6. Install libraries: `pip install -r requirements-dev.txt`
7. Check that `dataset/diabetes.csv` exists.
8. Train: `python train_model.py`
9. Start the server: `python app.py`
10. Open http://127.0.0.1:5000 in your browser. Press `Ctrl + C` in the terminal to stop.

## What each file does

| File | Job |
|------|-----|
| `dataset/diabetes.csv` | The data used to teach the model |
| `train_model.py` | Reads data, trains 2 models, compares them, saves the best |
| `model/*.pkl` | Saved model, scaler and imputer so the app does not retrain each time |
| `model/model_info.json` | Column order and scores |
| `model/deploy_model.json` | Small numpy-only copy of the model, used by the web app |
| `predictor.py` | Reads `deploy_model.json` and predicts using only NumPy |
| `requirements.txt` | Libraries the deployed app needs (Flask, NumPy) |
| `requirements-dev.txt` | Everything for training and the notebook |
| `vercel.json` | Tells Vercel which folders to include |
| `app.py` | Flask server: shows the page and answers `/predict` |
| `templates/index.html` | The web page structure |
| `public/static/style.css` | Colors, layout, mobile design |
| `public/static/script.js` | Checks inputs, calls Flask, shows result |
| `test_api.py` | Sends sample inputs to the API |
| `notebooks/diabetes_analysis.ipynb` | Data exploration with charts |

## How the pieces connect

1. You type values and click **Predict Diabetes**.
2. `script.js` checks them and sends JSON to `/predict` using `fetch()`.
3. `app.py` validates, then `predictor.py` fills empty values, scales them, and asks the model.
4. Flask sends back JSON with the result and probability.
5. `script.js` shows the result on the page, with no reload.

## The dataset

- 768 rows (patients), 9 columns. All patients are women of Pima Indian heritage, at least 21 years old.
- Inputs: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction (family-history score), Age.
- Target `Outcome`: 1 = diabetes, 0 = no diabetes.
- It suits **binary classification** because the answer has exactly two classes.
- **Problem zeros:** 5 Glucose, 35 BloodPressure, 227 SkinThickness, 374 Insulin and 11 BMI values are 0. A real person cannot have 0 glucose or 0 BMI, so these mean "not measured". We turn them into missing values and fill them with the median of the training data. This is why Skin Thickness and Insulin are optional in the form.
- **Limits:** The data is small and covers one population. Do not expect the model to work well for everyone.

## ML concepts in simple words

**Classification.** The model puts an input into a category. Here the categories are Diabetes (1) or No Diabetes (0), so it is *binary* classification.

**Logistic Regression.** It gives each input a weight, adds them up, and turns the total into a probability between 0 and 1. The **threshold** (usually 0.5) decides the label: probability >= 0.5 gives 1, otherwise 0.

**Random Forest.** A decision tree asks a chain of yes/no questions (for example "Glucose > 140?"). A Random Forest builds many trees on random parts of the data, and they **vote**. Combining many trees reduces mistakes, and it handles tabular data (rows and columns) well.

**Scaling.** Features have different ranges (Age 21-81, Insulin up to 846). Logistic Regression works better when features are on a similar scale, so we scale them. Random Forest does **not** need scaling, but scaling does not hurt it, so we use one pipeline for both.

**Train-test split.** If you test on the same data you trained on, the model can just memorise the answers and look perfect. We keep 20% of the data hidden and use it only for testing, to see how the model does on new data.

**Confusion matrix.**

| | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | True Negative (TN) | False Positive (FP) |
| **Actual Yes** | False Negative (FN) | True Positive (TP) |

- TP: has diabetes, model says diabetes. TN: no diabetes, model says no diabetes.
- FP: no diabetes, but model says diabetes (false alarm).
- FN: has diabetes, but model says no diabetes (missed case).

**Precision vs Recall.**
- Precision = TP / (TP + FP). When the model says "diabetes", how often is it right?
- Recall = TP / (TP + FN). Of all people who really have diabetes, how many did the model catch?
- In a health problem a **missed case (FN)** is usually worse than a false alarm, so recall matters a lot. That is why we do not judge by accuracy alone, and use `class_weight="balanced"` and F1-score.

## Testing

Run `python test_api.py` (the server need not be running). It sends 5 valid cases and several invalid ones. These are **software tests only, not medical predictions**.

Test the live API with curl (server running):

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"Pregnancies\":2,\"Glucose\":150,\"BloodPressure\":80,\"SkinThickness\":30,\"Insulin\":120,\"BMI\":32,\"DiabetesPedigreeFunction\":0.5,\"Age\":40}"
```

Also check http://127.0.0.1:5000/health in the browser.

Example values to type into the form:

| # | Preg | Glucose | BP | Skin | Insulin | BMI | DPF | Age |
|---|------|---------|----|------|---------|-----|-----|-----|
| 1 | 1 | 85 | 66 | 29 | (empty) | 26.6 | 0.351 | 31 |
| 2 | 8 | 183 | 64 | (empty) | (empty) | 33.3 | 0.672 | 52 |
| 3 | 3 | 120 | 72 | 25 | 100 | 30 | 0.4 | 35 |
| 4 | 0 | 90 | 70 | 20 | 80 | 22 | 0.2 | 22 |
| 5 | 5 | 160 | 85 | 35 | 200 | 38.5 | 0.9 | 58 |

## Common errors and fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'flask'` | Libraries not installed, or venv not active | Activate venv, then `pip install -r requirements-dev.txt` |
| `FileNotFoundError: Dataset not found` | `diabetes.csv` missing | Put it in the `dataset` folder |
| "Model files not found" when running `app.py` | You did not train yet | Run `python train_model.py` first |
| Flask will not start | Wrong folder or venv not active | `cd` into the project folder, activate venv, run `python app.py` |
| "Port 5000 is in use" | Another program uses it | Change `port=5000` to `5001` in `app.py`, or close the other program |
| Wrong number of features | Order or count of inputs changed | Keep the 8 fields in `FEATURES`; retrain if you change them |
| "must be a number" message | Text typed in a number field | Enter digits only, like `28.5` |
| CORS error | Only if the frontend is served from a different address | Not needed here, since Flask serves the page itself. If you separate them, run `pip install flask-cors` and add `CORS(app)` |
| Model loading error after upgrading scikit-learn | Saved model made with another version | Run `python train_model.py` again |
| `python` not found (Mac/Linux) | Command is named differently | Use `python3` |

## Upload to GitHub

1. Create a free account at github.com and install Git from git-scm.com.
2. On GitHub click **New repository**, name it `diabetes-prediction`, do **not** add a README (we have one), click Create.
3. In the project folder run:

```bash
git init
git add .
git commit -m "Initial commit: diabetes prediction web app"
git branch -M main
git remote add origin https://github.com/<your-username>/diabetes-prediction.git
git push -u origin main
```

`.gitignore` already blocks `venv/`, `__pycache__/`, `.env` and editor files. Check with `git status` before committing. The model files are only about 1.2 MB, so they are fine to upload. Never upload passwords or API keys.

Add screenshots: take them of the running app, put them in a `screenshots/` folder, and link them in the README.

## Deploy on Vercel (free public link)

1. Make sure the latest code is on GitHub.
2. Go to vercel.com and sign up with your GitHub account.
3. Click **Add New -> Project**, find `Diabetes-prediction`, click **Import**.
4. Leave all settings as they are and click **Deploy**.
5. After about a minute you get a link like `https://diabetes-prediction-xxxx.vercel.app`. Anyone can open it.
6. Every time you push to GitHub, Vercel redeploys automatically.

Why `predictor.py` exists: Vercel has a size limit, and scikit-learn + SciPy are too big. So `train_model.py` saves the trained model as `model/deploy_model.json`, and the web app predicts with only NumPy. It gives the same answers as scikit-learn (the training script checks this for you).

If you retrain the model, upload the new files in `model/` to GitHub again.

## Optional improvements (do these only after the basic version works)

1. Model comparison dashboard (show the metrics table on the page)
2. Feature importance chart
3. Prediction history
4. Stricter input validation
5. Downloadable PDF report
6. Deploy somewhere else, such as Render (add `gunicorn` to requirements, start command `gunicorn app:app`)
