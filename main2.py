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
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Real-time input for predictions
def predict_heart_disease():
    print("Enter the following details for heart disease prediction:")
    try:
        input_data = {
            'age': float(input("Age: ")),
            'sex': float(input("Sex (1=Male, 0=Female): ")),
            'chest_pain_type': float(input("Chest Pain Type (0-3): ")),
            'resting_blood_pressure': float(input("Resting Blood Pressure: ")),
            'cholesterol': float(input("Cholesterol Level: ")),
            'fasting_blood_sugar': float(input("Fasting Blood Sugar (1 for >120mg/dl, 0 otherwise): ")),
            'resting_electrocardiogram': float(input("Resting Electrocardiogram (0-2): ")),
            'max_heart_rate_achieved': float(input("Max Heart Rate Achieved: ")),
            'exercise_induced_angina': float(input("Exercise-Induced Angina (1 for Yes, 0 for No): ")),
            'st_depression': float(input("ST Depression Induced by Exercise: ")),
            'st_slope': float(input("ST Slope (0-2): ")),
            'ca': float(input("Number of Major Vessels (0-3): ")),
            'thal': float(input("Thalassemia (1-3): "))
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
            print("Prediction: The person is likely to have heart disease.")
        else:
            print("Prediction: The person is unlikely to have heart disease.")

    except Exception as e:
        print("Error in input. Please enter valid numeric values.", e)

# Call the prediction function in real-time
if __name__ == "__main__":
    predict_heart_disease()
