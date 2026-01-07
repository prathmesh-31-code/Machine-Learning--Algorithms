import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

# Load dataset
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv('diabetes.csv')

# Features and Target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_class = (lr_pred >= 0.5).astype(int)

# Random Forest Regression
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_class = (rf_pred >= 0.5).astype(int)

# Evaluation Function
def evaluate(name, y_true, y_prob, y_pred):
    print(f"{name} Results:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print(f"ROC AUC: {roc_auc_score(y_true, y_prob):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print()

# Evaluate both
evaluate("Linear Regression", y_test, lr_pred, lr_class)
evaluate("Random Forest Regressor", y_test, rf_pred, rf_class)

# Plot ROC Curve
fpr1, tpr1, _ = roc_curve(y_test, lr_pred)
fpr2, tpr2, _ = roc_curve(y_test, rf_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, label='Linear Regression')
plt.plot(fpr2, tpr2, label='Random Forest Regressor')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()
