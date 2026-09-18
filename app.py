from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the exported model and scaler
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

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
            result_text = f"1 Positive (Diabetic) with {probability * 100:.2f}% confidence."
        else:
            result_text = f"✅ Negative (Non-Diabetic) with {(1 - probability) * 100:.2f}% confidence."

        return render_template('index.html', prediction_text=result_text)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    # Enable debug mode to expose errors on the UI
    app.run(port=5000, debug=True, use_reloader=False)
