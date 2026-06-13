from flask import Flask, render_template, request
import joblib
app = Flask(__name__)

# Load trained model
model = joblib.load("diabetes_model.pkl")

print("=" * 50)
print("MODEL LOADED SUCCESSFULLY")
print("=" * 50)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    pregnancies = float(request.form["pregnancies"])
    glucose = float(request.form["glucose"])
    blood_pressure = float(request.form["blood_pressure"])
    skin_thickness = float(request.form["skin_thickness"])
    insulin = float(request.form["insulin"])
    bmi = float(request.form["bmi"])
    dpf = float(request.form["dpf"])
    age = float(request.form["age"])

    # ==========================
    # FEATURE ENGINEERING
    # ==========================

    if bmi < 18.5:
        bmi_category = 0
    elif bmi < 25:
        bmi_category = 1
    elif bmi < 30:
        bmi_category = 2
    else:
        bmi_category = 3

    if age < 30:
        age_group = 0
    elif age < 45:
        age_group = 1
    else:
        age_group = 2

    high_glucose = 1 if glucose > 140 else 0

    insulin_glucose_ratio = insulin / (glucose + 1)

    metabolic_risk = (
        0.4 * bmi +
        0.4 * glucose +
        0.2 * age
    )

    bp_bmi_risk = blood_pressure * bmi

    age_glucose_risk = age * glucose

    pregnancy_risk = pregnancies * age

    features = [[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age,
        bmi_category,
        age_group,
        high_glucose,
        insulin_glucose_ratio,
        metabolic_risk,
        bp_bmi_risk,
        age_glucose_risk,
        pregnancy_risk
    ]]
    print("Features Sent To Model:")
    print(features)

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1]

    risk_percentage = round(probability * 100, 2)

    if prediction == 1:
        result = "Diabetic"
    else:
        result = "Non-Diabetic"

    return render_template(
        "result.html",
        result=result,
        probability=risk_percentage
    )
if __name__ == "__main__":
    app.run(debug=True)