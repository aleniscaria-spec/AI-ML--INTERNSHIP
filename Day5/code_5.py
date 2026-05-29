from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import joblib as jb

data = pd.read_csv("Day5/Decision_tree.csv")
 
encoder_temp = LabelEncoder()
encoder_vib = LabelEncoder()
encoder_fail = LabelEncoder()
data["Temperature"] = encoder_temp.fit_transform(data["Temperature"])             #Encoding data
data["Vibration"] = encoder_vib.fit_transform(data["Vibration"])
data["Failure"] = encoder_fail.fit_transform(data["Failure"])
x = data[["Temperature", "Vibration"]]
y = data["Failure"]

model = DecisionTreeClassifier(criterion="entropy")
model.fit(x, y)
jb.dump(model, "machine_failure_Dtree_model.pkl")

new_data = pd.DataFrame({
    "Temperature": [encoder_temp.transform(["High"])[0]],
    "Vibration": [encoder_vib.transform(["Medium"])[0]]
})
prediction = model.predict(new_data)
result = encoder_fail.inverse_transform(prediction)

print("Encoded Dataset:\n")
print(data)
print("\nTemperature Encoding:")
for i, label in enumerate(encoder_temp.classes_):
    print(label, "=", i)
print("\nVibration Encoding:")
for i, label in enumerate(encoder_vib.classes_):
    print(label, "=", i)

print("\nFailure Encoding:")
for i, label in enumerate(encoder_fail.classes_):
    print(label, "=", i)

plt.figure(figsize=(14,8))
tree.plot_tree(
    model,
    feature_names=["Temperature", "Vibration"],
    class_names=["No Failure", "Failure"],
    filled=True,
    rounded=True,
    fontsize=11
)
plt.title("Decision Tree for Machine Failure Prediction",
          fontsize=16,
          fontweight="bold")

plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")

print("\nPrediction for New Machine:")
print("Temperature = High")
print("Vibration = Medium")
print("========================================")     
print("DECISION TREE RESULT")
print("========================================")
if prediction[0] == 1:
   print("Predicted Result: Failure of machine will occur")
else:
    print("Predicted Result: No Failure")
 
print("========================================")
plt.show()
