import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
data = pd.read_csv("Day3/KNN_Dataset.csv")

x = data[["Temperature"]]
y = data["Fuel_Consumption"]

model = KNeighborsRegressor(n_neighbors=3)
model.fit(x, y)
new_data = pd.DataFrame({"Temperature": [58]})
prediction = model.predict(new_data)

print("========================================")
print("KNN REGRESSION RESULT")
print("========================================")
print("Predicted fuel consumption for a temperature of 58 degrees is:", prediction[0], "liters per hour")
print("========================================")