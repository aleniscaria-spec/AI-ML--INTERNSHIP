import pandas as pd
import joblib  

best_model = joblib.load("insurance_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

age = int(input("Enter the age of primary beneficiary: "))
while True:
    sex = input("Sex (male/female): ").lower()
    if sex in ["male", "female"]:
        break
    print("Invalid input! Please enter male or female.") 

bmi = float(input("Enter the BMI: "))
children = int(input("Enter the number of children: "))

while True:
    smoker = input("Is the beneficiary a smoker? (yes/no): ").lower()
    if smoker in ["yes", "no"]:
        break
    print("Invalid input! Please enter yes or no.")
while True:
    region = input("Enter the region (southwest/southeast/northwest/northeast): ").lower()
    if region in ["southwest", "southeast", "northwest", "northeast"]:
        break
    print("Invalid input! Please enter a valid region.")

sex = encoders["sex"].transform([sex])[0]
smoker = encoders["smoker"].transform([smoker])[0]
region = encoders["region"].transform([region])[0]

new_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region] })

data_scaled = scaler.transform(new_data)
prediction = best_model.predict(data_scaled)
print('-' * 40)
print(f"\nPredicted Insurance Charge: ${prediction[0]:,.2f}")
print('-' * 40)