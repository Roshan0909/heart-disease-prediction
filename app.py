from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset
data = pd.read_csv('heart.csv')

# Data cleaning
data = data[(data['ca'] < 4) & (data['thal'] > 0)]

# Rename columns for clarity
data.rename(columns={
    'cp': 'chest_pain_type',
    'trestbps': 'resting_blood_pressure',
    'chol': 'cholesterol',
    'fbs': 'fasting_blood_sugar',
    'restecg': 'resting_electrocardiogram',
    'thalach': 'max_heart_rate_achieved',
    'exang': 'exercise_induced_angina',
    'oldpeak': 'st_depression',
    'slope': 'st_slope'
}, inplace=True)

# Prepare features and target
X = data.drop(columns=['target'])
y = data['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Collect input data
            input_data = [
                float(request.form['age']),
                int(request.form['sex']),
                int(request.form['chest_pain_type']),
                float(request.form['resting_blood_pressure']),
                float(request.form['cholesterol']),
                int(request.form['fasting_blood_sugar']),
                int(request.form['resting_electrocardiogram']),
                float(request.form['max_heart_rate_achieved']),
                int(request.form['exercise_induced_angina']),
                float(request.form['st_depression']),
                int(request.form['st_slope']),
                int(request.form['ca']),
                int(request.form['thal'])
            ]

            # Process input
            input_features = np.array(input_data).reshape(1, -1)
            input_features = scaler.transform(input_features)

            # Make prediction
            prediction = model.predict(input_features)[0]
            prediction_text = "Likely to have heart disease" if prediction == 1 else "Unlikely to have heart disease"

            return render_template('result.html', prediction=prediction_text)

        except Exception as e:
            return render_template('result.html', prediction=f"Error: {e}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
