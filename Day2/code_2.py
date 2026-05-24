import pandas as pd
from sklearn.linear_model import LinearRegression  

wind_speed = float(input("Enter Wind Speed: "))
blade_angle = float(input("Enter Blade Angle: "))
rotor_speed = float(input("Enter Rotor Speed: "))
new_data = [[wind_speed, blade_angle, rotor_speed]]

data = pd.read_csv("Day2/power_data.csv")
x = data[["Wind_Speed","Blade_Angle","Rotor_Speed"]]
y = data["Power_Output"]

model = LinearRegression()
model.fit(x, y)

slope = model.coef_
intercept = model.intercept_

print(f"Regression Equation: y ={slope}x + {intercept} ")


predicted = model.predict(new_data)
predicted_print =-1*predicted  

print("Coefficient for Wind Speed:", slope[0])
print("Coefficient for Blade Angle:", slope[1])
print("Coefficient for Rotor Speed:", slope[2])
print("Intercept of the regression line:",intercept)
print(f"Regression Equation: {slope[0]:.4f}*x_1 + {slope[1]:.4f}*x_2 + {slope[2]:.4f}*x_3  {intercept:.4f}")
print("Predicted power output for the given data is :", predicted[0],'kW')