import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data = pd.read_csv("Day1/Load_Extension.csv")

data.columns = data.columns.str.strip()

x = data[['Load']]
y = data['Extension']

model = LinearRegression()
model.fit(x, y)

slope = model.coef_[0]
intercept = model.intercept_

print("Slope:", slope)
print("Intercept:", intercept)

print(f"Regression Equation: y ={slope:.4f}x + {intercept:.4f} ")

predicted = model.predict(pd.DataFrame({'Load':[55]}))

print("Predicted extension for 55 N:", predicted[0])

y_pred = model.predict(x)

plt.scatter(x, y, label='Observed Data')
plt.plot(x, y_pred, label='Regression Line')
plt.xlabel("Load (N)")
plt.ylabel("Extension (mm)")
plt.title("Load vs Spring Extension")
plt.legend()

plt.grid(True)
plt.show()
