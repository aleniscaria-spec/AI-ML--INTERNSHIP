from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib

data = pd.read_csv(r"C:\Users\Aleni\OneDrive\Documents\AI_ML_INTERNSHIP\Day42\Breast_cancer_data.csv")                #data sorting
data = data.drop(["id", "Unnamed: 32"], axis=1)

  

data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})


X = data.drop("diagnosis", axis=1)                              #Assigning the data to X and Y
y = data["diagnosis"]


X_train, X_test, y_train, y_test = train_test_split(            #splitting data into training and testing sets
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=10000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
joblib.dump(model, "breast_cancer_prediction_model.pkl")
user_input = []
print("\nEnter tumor measurement values:\n")

for column in X.columns:                                       #using for loop for getting user input
    value = float(input(f"Enter {column}: "))
    user_input.append(value)

new_data = pd.DataFrame([user_input], columns=X.columns)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


prediction = model.predict(new_data)

print("========================================")
if prediction[0] == 1:
    print("\nPrediction: Malignant Tumor (Cancerous)")
else:
    print("\nPrediction: Benign Tumor (Non-Cancerous)")
print("========================================")                        
