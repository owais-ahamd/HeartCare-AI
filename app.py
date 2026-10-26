from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__, template_folder="templates")

model = joblib.load("models/heart_model.pkl")
scaler = joblib.load("models/scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    values = [
        float(request.form["age"]),
        float(request.form["sex"]),
        float(request.form["cp"]),
        float(request.form["trestbps"]),
        float(request.form["chol"]),
        float(request.form["fbs"]),
        float(request.form["restecg"]),
        float(request.form["thalach"]),
        float(request.form["exang"]),
        float(request.form["oldpeak"]),
        float(request.form["slope"]),
        float(request.form["ca"]),
        float(request.form["thal"])
    ]

    input_data = np.array(values).reshape(1, -1)

    input_scaled = scaler.transform(input_data)

    prediction_value = model.predict(input_scaled)[0]

    probabilities = model.predict_proba(input_scaled)[0]

    class_index = list(model.classes_).index(1)

    probability = probabilities[class_index] * 100

    if prediction_value == 1:
        prediction = "Heart Disease Detected"
    else:
        prediction = "No Heart Disease"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=round(probability, 2)
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )