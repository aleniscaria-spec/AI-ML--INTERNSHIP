from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pandas as pd
import joblib as jb
import numpy as np

data = pd.read_csv("Mini_Project_1/Concrete_data.csv")
data = data.drop_duplicates()
#print(data.columns)
#data preprocessing
x = data.drop('Concrete compressive strength', axis=1)
y = data["Concrete compressive strength"]
#print(data.head())
#print(data.shape)
#print(data.isnull().sum())
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
'''
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)'''

linear = LinearRegression()
linear.fit(x_train_scaled, y_train)

linear_pred = linear.predict(x_test_scaled)
linear_r2 = r2_score(y_test, linear_pred)
linear_mae = mean_absolute_error(y_test, linear_pred)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))

best_knn = None
best_knn_r2 = float("-inf")
best_k = 0

for i in range(1, 21,1):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(x_train_scaled, y_train)
    pred = knn.predict(x_test_scaled)
    r2 = r2_score(y_test, pred)
    if r2 > best_knn_r2:
        best_knn_r2 = r2
        best_knn = knn
        best_k = i

knn_pred = best_knn.predict(x_test_scaled)
knn_mae = mean_absolute_error(y_test, knn_pred)
knn_rmse = np.sqrt(mean_squared_error(y_test, knn_pred))

best_rf = None
best_rf_r2 = float("-inf")
best_estimators = 0

for j in range(20, 50, 2):
    rf = RandomForestRegressor(
        n_estimators=j,
        random_state=42)
    rf.fit(x_train_scaled, y_train)
    pred = rf.predict(x_test_scaled)
    r2 = r2_score(y_test, pred)
    if r2 > best_rf_r2:
        best_rf_r2 = r2
        best_rf = rf
        best_estimators = j

rf_pred = best_rf.predict(x_test_scaled)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print("\nMODEL COMPARISON")
print("-" * 40)
print("\nLinear Regression")
print(f"R² Score : {linear_r2:.4f}")
print(f"MAE      : {linear_mae:.2f}")
print(f"RMSE     : {linear_rmse:.2f}")
print('-' * 40)
print(f"\nBest KNN (k={best_k})")
print(f"R² Score : {best_knn_r2:.4f}")
print(f"MAE      : {knn_mae:.2f}")
print(f"RMSE     : {knn_rmse:.2f}")
print('-' * 40)
print(f"\nBest Random Forest (n_estimators={best_estimators})")
print(f"R² Score : {best_rf_r2:.4f}")
print(f"MAE      : {rf_mae:.2f}")
print(f"RMSE     : {rf_rmse:.2f}")
print('-' * 40)
model = {
    "Linear Regression": (linear, linear_r2),
    "KNN": (best_knn, best_knn_r2),
    "Random Forest": (best_rf, best_rf_r2)}

best_model_name = max(model, key=lambda x: model[x][1])
best_model = model[best_model_name][0]
print(f"\nBest Overall Model: {best_model_name}")
print('-' * 40)
jb.dump(best_model, "best_concrete_model.pkl")
jb.dump(scaler, "concrete_scaler.pkl")
