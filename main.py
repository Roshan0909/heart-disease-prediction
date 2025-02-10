import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data = pd.read_csv('heart.csv')

# Clean the dataset
data = data[data['ca'] < 4]  # Drop invalid 'ca' values
data = data[data['thal'] > 0]  # Drop invalid 'thal' values

# Rename columns for clarity
data = data.rename(
    columns={
        'cp': 'chest_pain_type',
        'trestbps': 'resting_blood_pressure',
        'chol': 'cholesterol',
        'fbs': 'fasting_blood_sugar',
        'restecg': 'resting_electrocardiogram',
        'thalach': 'max_heart_rate_achieved',
        'exang': 'exercise_induced_angina',
        'oldpeak': 'st_depression',
        'slope': 'st_slope'
    }
)

# Prepare features and target
X = data.drop('target', axis=1)
y = data['target']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
# st.write("### Model Evaluation Results")
# st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
# st.write("Classification Report:\n", classification_report(y_test, y_pred))

# Streamlit app UI for prediction
st.title("Heart Disease Prediction")

st.write("Please enter the following details to predict heart disease:")

# Create a form for user input
with st.form(key="prediction_form"):
    age = st.number_input("Age:", min_value=20, max_value=120, step=1, value=50, help="Example: 50")
    sex = st.selectbox("Sex (1=Male, 0=Female):", options=[0, 1], help="Select 1 for Male and 0 for Female")
    chest_pain_type = st.slider("Chest Pain Type (0-3):", min_value=0, max_value=3, step=1, value=1, help="Example: 1")
    resting_blood_pressure = st.number_input("Resting Blood Pressure (in mm Hg):", min_value=80, max_value=200, step=1, value=120, help="Example: 120")
    cholesterol = st.number_input("Cholesterol Level:", min_value=100, max_value=600, step=1, value=200, help="Example: 200")
    fasting_blood_sugar = st.selectbox("Fasting Blood Sugar (1 for >120mg/dl, 0 otherwise):", options=[0, 1], help="Select 1 if blood sugar >120mg/dl, else 0")
    resting_electrocardiogram = st.slider("Resting Electrocardiogram (0-2):", min_value=0, max_value=2, step=1, value=0, help="Example: 0")
    max_heart_rate_achieved = st.number_input("Max Heart Rate Achieved:", min_value=60, max_value=200, step=1, value=150, help="Example: 150")
    exercise_induced_angina = st.selectbox("Exercise-Induced Angina (1 for Yes, 0 for No):", options=[0, 1], help="Select 1 for Yes and 0 for No")
    st_depression = st.number_input("ST Depression Induced by Exercise:", min_value=0.0, max_value=6.0, step=0.1, value=1.0, help="Example: 1.0")
    st_slope = st.slider("ST Slope (0-2):", min_value=0, max_value=2, step=1, value=1, help="Example: 1")
    ca = st.slider("Number of Major Vessels (0-3):", min_value=0, max_value=3, step=1, value=0, help="Example: 0")
    thal = st.slider("Thalassemia (1-3):", min_value=1, max_value=3, step=1, value=2, help="Example: 2")

    submit_button = st.form_submit_button(label="Predict")

# Make a prediction when the form is submitted
if submit_button:
    try:
        input_data = {
            'age': age,
            'sex': sex,
            'chest_pain_type': chest_pain_type,
            'resting_blood_pressure': resting_blood_pressure,
            'cholesterol': cholesterol,
            'fasting_blood_sugar': fasting_blood_sugar,
            'resting_electrocardiogram': resting_electrocardiogram,
            'max_heart_rate_achieved': max_heart_rate_achieved,
            'exercise_induced_angina': exercise_induced_angina,
            'st_depression': st_depression,
            'st_slope': st_slope,
            'ca': ca,
            'thal': thal
        }

        # Create a feature array in the correct order
        feature_order = X.columns  # Ensures input matches training data structure
        input_features = np.array([input_data[feature] for feature in feature_order]).reshape(1, -1)

        # Convert input to DataFrame to suppress warnings and standardize it
        input_features = pd.DataFrame(input_features, columns=feature_order)
        input_features = scaler.transform(input_features)

        # Make a prediction
        prediction = model.predict(input_features)
        if prediction[0] == 1:
            st.write("### Prediction: The person is likely to have heart disease.")
        else:
            st.write("### Prediction: The person is unlikely to have heart disease.")

    except Exception as e:
        st.error(f"Error in input. Please enter valid numeric values. ({e})")
