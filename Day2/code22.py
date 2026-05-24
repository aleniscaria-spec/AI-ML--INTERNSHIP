import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Day2/power_data.csv")

x=data[["Wind_Speed","Blade_Angle","Rotor_Speed"]]
y=data["Power_Output"]

model = LinearRegression()
model.fit(x,y)

print("Coefficient: ", model.coef_)
print("Intercept: ", model.intercept_)

newData = [[9,11,100]]
new_power = model.predict(newData)
print(f"New Power output = {new_power} kW")