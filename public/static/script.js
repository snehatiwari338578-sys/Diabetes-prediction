// script.js
// Reads the form, checks the numbers, sends them to Flask with fetch(),
// and shows the answer on the page without reloading it.

// Rules for each field: min, max, and whether it can be left empty.
const FIELDS = {
  Pregnancies:              { label: "Pregnancies",                min: 0,    max: 20,  optional: false },
  Glucose:                  { label: "Glucose",                    min: 40,   max: 300, optional: false },
  BloodPressure:            { label: "Blood Pressure",             min: 30,   max: 200, optional: false },
  SkinThickness:            { label: "Skin Thickness",             min: 5,    max: 100, optional: true  },
  Insulin:                  { label: "Insulin",                    min: 10,   max: 900, optional: true  },
  BMI:                      { label: "BMI",                        min: 10,   max: 70,  optional: false },
  DiabetesPedigreeFunction: { label: "Diabetes Pedigree Function", min: 0.05, max: 2.5, optional: false },
  Age:                      { label: "Age",                        min: 1,    max: 120, optional: false },
};

// Grab page elements once
const form = document.getElementById("predict-form");
const predictBtn = document.getElementById("predict-btn");
const resetBtn = document.getElementById("reset-btn");
const resultCard = document.getElementById("result-card");
const resultText = document.getElementById("result-text");
const resultProb = document.getElementById("result-prob");
const resultNote = document.getElementById("result-note");
const alertBox = document.getElementById("alert-box");

// Show or clear an error under one input
function setFieldError(name, message) {
  document.getElementById("err-" + name).textContent = message;
  document.getElementById(name).classList.toggle("invalid", message !== "");
}

// Check every input. Returns the data object, or null if something is wrong.
function validateForm() {
  const data = {};
  let allGood = true;

  for (const name in FIELDS) {
    const rule = FIELDS[name];
    const raw = document.getElementById(name).value.trim();
    setFieldError(name, "");

    if (raw === "") {
      if (rule.optional) {
        data[name] = null; // backend will fill it in
        continue;
      }
      setFieldError(name, rule.label + " is required.");
      allGood = false;
      continue;
    }

    const number = Number(raw);
    if (Number.isNaN(number)) {
      setFieldError(name, "Please enter a valid number.");
      allGood = false;
    } else if (rule.optional && number === 0) {
      data[name] = null; // 0 means "not measured" for optional fields
    } else if (number < rule.min || number > rule.max) {
      setFieldError(name, "Enter a value between " + rule.min + " and " + rule.max + ".");
      allGood = false;
    } else {
      data[name] = number;
    }
  }
  return allGood ? data : null;
}

function showAlert(message) {
  alertBox.textContent = message;
  alertBox.classList.remove("hidden");
}

function hideMessages() {
  alertBox.classList.add("hidden");
  resultCard.classList.add("hidden");
}

// When the form is submitted
form.addEventListener("submit", async function (event) {
  event.preventDefault(); // stop the page from reloading
  hideMessages();

  const data = validateForm();
  if (data === null) return; // errors are already shown under the inputs

  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting...";

  try {
    // Send the data to Flask
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const result = await response.json();

    if (!response.ok || !result.success) {
      showAlert(result.error || "Something went wrong. Please try again.");
    } else {
      showResult(result);
    }
  } catch (err) {
    // Happens if the server is off or the network fails
    showAlert("Could not reach the server. Please check that Flask is running.");
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict Diabetes";
  }
});

// Display the prediction
function showResult(result) {
  resultCard.classList.remove("positive", "negative");
  resultCard.classList.add(result.prediction === 1 ? "positive" : "negative");

  resultText.textContent = "Prediction: " + result.result;
  resultProb.textContent = "Model probability estimate: " + result.probability + "%";
  resultNote.textContent =
    "This is the model's estimate (" + result.model + "), not your actual medical risk. " +
    "It is an educational project and not a medical diagnosis.";

  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// Reset button clears everything
resetBtn.addEventListener("click", function () {
  form.reset();
  for (const name in FIELDS) setFieldError(name, "");
  hideMessages();
});
