import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the diabetes dataset
diabetes_data = pd.read_csv("diabetes.csv")  # Ensure correct file name

# Define features (X) and target (y)
X = diabetes_data.drop(columns=["Outcome"])  # Ensure Outcome is the correct target column
y = diabetes_data["Outcome"]

# Define Stratified K-Fold
k = 5  # Number of folds
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Store accuracy for each fold
fold_accuracies = []
all_reports = []

# Perform K-Fold Cross-Validation
for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=5)  # Tune k as needed
    knn.fit(X_train_scaled, y_train)

    # Predict
    y_pred = knn.predict(X_test_scaled)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    fold_accuracies.append(accuracy)

    # Store classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    all_reports.append(report)

    # Print individual fold results
    print(f"Fold {fold} Accuracy: {accuracy:.4f}")
    print(f"Fold {fold} Classification Report:\n", classification_report(y_test, y_pred))

# Print overall K-Fold performance
print(f"\nAverage Accuracy across {k} folds: {np.mean(fold_accuracies):.4f}")
