from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd
import joblib

data = pd.read_csv("Day8/heart.csv")
x = data.drop("target", axis=1)
y = data["target"]

thal = LabelEncoder()
x["thal"] = thal.fit_transform(x["thal"])
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2,random_state=42,)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = Sequential([
    Dense(16, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit( x_train_scaled, y_train, epochs=500, batch_size=150, validation_split=0.2, verbose=1)

y_pred_prob = model.predict(x_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)
print('*' *40)
print("Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('*' *40)
model.save("heart_ann_model.keras")
joblib.dump(scaler, "heart_scaler.pkl")
joblib.dump(thal, "heart_thal_encoder.pkl")

print("Model saved successfully!")
print("Scaler saved successfully!")
print("Thal Encoder saved successfully!")
print('*' *40)