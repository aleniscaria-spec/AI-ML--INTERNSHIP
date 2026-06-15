# Concrete Compressive Strength Prediction Using Machine Learning

## Overview

This project uses Machine Learning to predict the compressive strength of concrete based on its material composition and curing age. The goal is to develop a regression model that can estimate concrete strength accurately, helping engineers optimize concrete mix designs.

## Dataset

The project uses the Concrete Compressive Strength dataset containing 1005 records with the following features:

* Cement
* Blast Furnace Slag
* Fly Ash
* Water
* Superplasticizer
* Coarse Aggregate
* Fine Aggregate
* Age

**Target Variable:**

* Concrete Compressive Strength (MPa)

Dataset Source:
https://www.kaggle.com/datasets/niteshyadav3103/concrete-compressive-strength

---

## Project Workflow

1. Data Loading using Pandas
2. Missing Value Analysis
3. Feature and Target Separation
4. Feature Scaling using StandardScaler
5. Train-Test Split (80:20)
6. Model Training using Random Forest Regressor
7. Model Evaluation
8. Prediction for New Inputs

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

---

## Model Performance

| Metric   | Value  |
| -------- | ------ |
| MAE      | 3.12   |
| RMSE     | 4.96   |
| R² Score | 0.9174 |

The model achieved an R² score of 0.9174, indicating strong predictive performance and the ability to explain over 91% of the variance in concrete compressive strength.

---

## Visualizations

The project includes:

* Correlation Heatmap
* Actual vs Predicted Plot
* Feature Importance Analysis

---

## Files

* `Concre.py` – Model training script
* `concrete_input.py` – Prediction script using saved model
* `concrete_model.pkl` – Trained Random Forest model
* `scaler.pkl` – Saved StandardScaler object
* `CONCRETE COMPRESSIVE STRENGTH.pdf` – Project report

---

## Future Improvements

* Development of a graphical user interface
* Deployment as a web application

---