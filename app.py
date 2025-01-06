from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define columns
columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
           'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
           'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
           'InternetService_DSL', 'InternetService_Fiber optic',
           'InternetService_No', 'Contract_Month-to-month', 'Contract_One year',
           'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)',
           'PaymentMethod_Credit card (automatic)',
           'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    user_input = [request.form[col] for col in columns]
    user_input = np.array(user_input, dtype=float).reshape(1, -1)

    # Create DataFrame with correct column names
    input_df = pd.DataFrame(user_input, columns=columns)

    # Predict
    prediction = model.predict(input_df)[0]

    result = 'Churn: Yes' if prediction > 0.5 else 'Churn: No'
    return render_template('index.html', prediction_text=f'Result: {result}')


if __name__ == '__main__':
    app.run(debug=True)
