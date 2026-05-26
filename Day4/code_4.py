from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib

data = pd.read_csv("Day4/machine_failure.csv")

data = data.drop(["UDI", "Product ID"], axis=1)
encoder = LabelEncoder()
data["Type"] = encoder.fit_transform(data["Type"])

x = data.drop("Machine failure", axis=1)
y = data["Machine failure"] 


model = LogisticRegression(max_iter=1000)
model.fit(x, y)
joblib.dump(model, "machine_failure_model.pkl")

sample_type = encoder.transform(["M"])[0]

new_data = pd.DataFrame({
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

prediction = model.predict(new_data)


print("========================================")
print("LOGISTIC REGRESSION RESULT")
print("========================================")
if prediction[0] == 1:
   print("Predicted Result: Failure of machine will occur")
else:
    print("Predicted Result: No Failure")
 
print("========================================")