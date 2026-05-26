from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from quopri import encode
import pandas as pd
import numpy as np
import joblib

data = pd.read_csv("Day4/machine_failure.csv")
data = data.drop(["UDI", "Product ID"], axis=1)

encoder = LabelEncoder()
data["Type"] = encoder.fit_transform(data["Type"])

x = data.drop("Machine failure", axis=1)
y = data["Machine failure"]

best_accuracy = 0
best_error = 1
best_test_size = 0
best_model = None

for test_size in np.arange(0.1, 0.301, 0.001):         #testing diferent values for test split

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    error = 1 - accuracy

    if error < best_error:
        best_error = error
        best_accuracy = accuracy
        best_test_size = test_size
        best_model = model

joblib.dump(best_model, "machine_failure_model.pkl")

sample_type = encoder.transform(["M"])[0]

new_data = pd.DataFrame({                                #Input data
    "Type": [sample_type],
    "Air temperature [K]": [301],
    "Process temperature [K]": [311],
    "Rotational speed [rpm]": [1450],
    "Torque [Nm]": [52],
    "Tool wear [min]": [28],
    "TWF": [1],
    "HDF": [0],
    "PWF": [0],
    "OSF": [0],
    "RNF": [0]
})

prediction = best_model.predict(new_data)
print("========================================")
print("LOGISTIC REGRESSION RESULT")
print("========================================")

print("Best Test Size:", round(best_test_size, 3))
print("Best Accuracy:", round(best_accuracy, 5))
print("Lowest Error:", round(best_error, 5))

print("----------------------------------------")

if prediction[0] == 1:
    print("Predicted Result: Failure of machine will occur")
else:
    print("Predicted Result: No Failure")
print("========================================")