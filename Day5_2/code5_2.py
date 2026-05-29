from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import pandas as pd
import joblib as jb
import matplotlib.pyplot as plt

data = pd.read_csv("Day5_2/DTree_2.csv")
x = data.drop(["fail", "footfall"], axis=1)
y = data["fail"]

x_train, x_test, y_train, y_test = train_test_split(          #splitting the dataset into training and testing sets
    x, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42,
    max_depth=4
)

model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
print("\nModel Accuracy:", accuracy)

new_data = pd.DataFrame({
    "tempMode": [4],
    "AQ": [5],
    "USS": [3],
    "CS": [6],
    "VOC": [1],
    "RP": [45],
    "IP": [5],
    "Temperature": [1]
})

prediction = model.predict(new_data)

print("\nPrediction Result:", prediction[0])
print("========================================")
print("DECISION TREE RESULT")
print("========================================")
if prediction[0] == 1:
   print("Predicted Result: Failure of machine will occur")
else:
    print("Predicted Result: No Failure")
 
print("========================================")
plt.figure(figsize=(25, 15))

tree.plot_tree(
    model,
    feature_names=x.columns,
    class_names=["No Failure", "Failure"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree Classifier", fontsize=20)
plt.show()