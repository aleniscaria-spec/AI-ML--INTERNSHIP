import pandas as pd
import joblib

model = joblib.load("Mini_Project_1/best_concrete_model.pkl")
scaler = joblib.load("Mini_Project_1/concrete_scaler.pkl")

cement = float(input("Enter Cement: "))
slag = float(input("Enter Blast Furnace Slag: "))
flyash = float(input("Enter Fly Ash: "))
water = float(input("Enter Water: "))
superplasticizer = float(input("Enter Superplasticizer: "))
coarseagg = float(input("Enter Coarse Aggregate: "))
fineagg = float(input("Enter Fine Aggregate: "))
age = float(input("Enter Age (days): "))

new_data = pd.DataFrame({
    'Cement': [cement],
    'Blast Furnace Slag': [slag],
    'Fly Ash': [flyash],
    'Water': [water],
    'Superplasticizer': [superplasticizer],
    'Coarse Aggregate': [coarseagg],
    'Fine Aggregate': [fineagg],
    'Age (day)':[age]
})

new_data_scaled = scaler.transform(new_data)


prediction = model.predict(new_data_scaled)
print('*'*40)
print(f"\nPredicted Concrete Strength: {prediction[0]:.2f} MPa")
print('*'*40)