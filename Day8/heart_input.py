import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("Day8/heart_ann_model.keras")
scaler = joblib.load("Day8/heart_scaler.pkl")
thal_encoder = joblib.load("Day8/heart_thal_encoder.pkl")

print("Enter Patient Details")

age = float(input("Age: "))
sex = int(input("Sex (0=Female, 1=Male): "))
cp = int(input("Chest Pain Type (0-3):" \
"0=Typical Angina, 1=Atypical Angina, 2=Non-Anginal Pain, 3=Asymptomatic): "))
trestbps = float(input("Resting Blood Pressure(Systole): "))
chol = float(input("Cholesterol:(Serum Cholesterol in mg/dl) "))
fbs = int(input("Fasting Blood Sugar (if greater than 120 mg/dl=1, otherwise 0): "))
restecg = int(input("Rest ECG (if normal=0,if ST-T wave abnormality=1,definite left ventricular hypertrophy=2): "))
thalach = float(input("Maximum Heart Rate: "))
exang = int(input("Exercise Angina (if present=1, otherwise 0): "))
oldpeak = float(input("Oldpeak(ST Depression induced by exercise relative to rest): "))
slope = int(input("Slope of the peak exercise ST segment (Upslope=1, Flat=2, Downslope=3): "))
ca = int(input("Number of major vessels (0-3) colored by fluoroscopy: "))
thal_text = input("Thal (fixed/normal/reversible): ").lower()
thal = thal_encoder.transform([thal_text])[0]

patient_data = np.array([[
    age, sex, cp, trestbps, chol, fbs, restecg, thalach,exang, oldpeak, slope,ca, thal]])
patient_scaled = scaler.transform(patient_data)
probability = model.predict(patient_scaled, verbose=0)[0][0]
print('*'*40)
if probability >= 0.5:
    print("\nHeart Disease Detected")
else:
    print("\nNo Heart Disease Detected")

print(f"Prediction Probability: {probability:.2%}")
print('*'*40)