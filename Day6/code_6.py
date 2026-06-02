from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

data = pd.read_csv("Day6/iris.csv")
data = data.drop(["Id"], axis=1)
x = data.drop("Species", axis=1)
y = data["Species"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
joblib.dump(model, "Species_prediction_model.pkl")
flower_codes = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}
print("\nModel Accuracy:", accuracy)
new_data = [
    [5.0, 3.6, 1.4, 0.2],  
    [6.1, 2.8, 4.0, 1.3],  
    [6.9, 3.1, 5.4, 2.1] ]

predictions = model.predict(new_data)
for sample, flower_name in zip(new_data, predictions):

    flower_code = flower_codes[flower_name]

    print("========================================")
    print("Input Measurements")
    print("========================================")

    print(f"Sepal Length : {sample[0]}")
    print(f"Sepal Width  : {sample[1]}")
    print(f"Petal Length : {sample[2]}")
    print(f"Petal Width  : {sample[3]}")

    print("\nPrediction Result")
    print(f"Flower Code : {flower_code}")
    print(f"Flower Name : {flower_name.upper()}")

    print("========================================")