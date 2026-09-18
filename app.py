from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load the exported model and scaler
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

# Self-contained HTML template string to avoid directory search bugs
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Prediction Portal</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; }
        .card { border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    <div class="container mt-5">
        <div class="row justify-content-center">
            <div class="col-md-8">
                <div class="card p-4">
                    <h2 class="text-center mb-4 text-primary">Diabetes Prediction Portal</h2>
                    <p class="text-muted text-center">Fill in the details below to predict the probability of diabetes.</p>

                    <form action="/predict" method="POST">
                        <div class="row">
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Pregnancies</label>
                                <input type="number" step="any" name="Pregnancies" class="form-control" value="3" required>
                            </div>
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Glucose</label>
                                <input type="number" step="any" name="Glucose" class="form-control" value="117" required>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Blood Pressure</label>
                                <input type="number" step="any" name="BloodPressure" class="form-control" value="72" required>
                            </div>
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Skin Thickness</label>
                                <input type="number" step="any" name="SkinThickness" class="form-control" value="23" required>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Insulin</label>
                                <input type="number" step="any" name="Insulin" class="form-control" value="30" required>
                            </div>
                            <div class="col-md-6 mb-3">
                                <label class="form-label">BMI</label>
                                <input type="number" step="any" name="BMI" class="form-control" value="32.0" required>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Diabetes Pedigree Function</label>
                                <input type="number" step="0.001" name="DiabetesPedigreeFunction" class="form-control" value="0.372" required>
                            </div>
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Age</label>
                                <input type="number" step="any" name="Age" class="form-control" value="29" required>
                            </div>
                        </div>
                        <button type="submit" class="btn btn-primary w-100 mt-3">Predict</button>
                    </form>

                    {% if prediction_text %}
                    <div class="alert alert-info mt-4 text-center fs-5">
                        {{ prediction_text }}
                    </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input from the HTML form
        features = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]

        # Prepare the input for prediction
        final_features = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(final_features)

        # Predict outcome and confidence level
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1]

        if prediction == 1:
            result_text = f"Positive (Diabetic) with {probability * 100:.2f}% confidence."
        else:
            result_text = f"Negative (Non-Diabetic) with {(1 - probability) * 100:.2f}% confidence."

        return render_template_string(HTML_TEMPLATE, prediction_text=result_text)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(port=5000, debug=True, use_reloader=False)
