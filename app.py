from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "heart_disease_model.joblib"
model = None

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️ Model not found: {e}")

# Feature options based on typical heart disease dataset
FEATURE_OPTIONS = {
    'age': list(range(29, 80)),
    'sex': [('1', 'Male'), ('0', 'Female')],
    'cp': [
        ('0', 'Typical Angina'),
        ('1', 'Atypical Angina'),
        ('2', 'Non-anginal Pain'),
        ('3', 'Asymptomatic')
    ],
    'trestbps': list(range(90, 201)),
    'chol': list(range(120, 565)),
    'fbs': [('1', 'True (>120 mg/dl)'), ('0', 'False (<=120 mg/dl)')],
    'restecg': [
        ('0', 'Normal'),
        ('1', 'ST-T Wave Abnormality'),
        ('2', 'Left Ventricular Hypertrophy')
    ],
    'thalach': list(range(70, 203)),
    'exang': [('1', 'Yes'), ('0', 'No')],
    'oldpeak': [round(x * 0.1, 1) for x in range(0, 63)],
    'slope': [
        ('0', 'Upsloping'),
        ('1', 'Flat'),
        ('2', 'Downsloping')
    ],
    'ca': [('0', '0'), ('1', '1'), ('2', '2'), ('3', '3'), ('4', '4')],
    'thal': [
        ('0', 'Normal'),
        ('1', 'Fixed Defect'),
        ('2', 'Reversible Defect'),
        ('3', 'Reversible Defect')
    ]
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html', features=FEATURE_OPTIONS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get form data
        data = request.form.to_dict()
        
        # Create DataFrame with feature engineering
        input_df = pd.DataFrame([data])
        
        # Convert to numeric
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col])
        
        # Add age_bin feature (matching training)
        if 'age' in input_df.columns:
            input_df['age_bin'] = pd.cut(
                input_df['age'], 
                bins=[0, 30, 40, 50, 60, 120], 
                labels=['<30', '30-40', '40-50', '50-60', '60+']
            )
        
        # Add interaction feature
        if 'age' in input_df.columns and 'chol' in input_df.columns:
            input_df['age_chol'] = input_df['age'] * input_df['chol']
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        result = {
            'prediction': int(prediction),
            'probability': {
                'no_disease': float(probability[0]),
                'disease': float(probability[1])
            },
            'risk_level': 'High Risk' if probability[1] > 0.7 else 'Moderate Risk' if probability[1] > 0.4 else 'Low Risk'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)