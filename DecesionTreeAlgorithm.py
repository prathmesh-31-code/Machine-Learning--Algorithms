import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the preprocessed data
data = pd.read_csv('cleaned_employee_data.csv')

# Display the first few rows of the preprocessed dataset
print(data.head())

# Ensure 'performance rating' is categorical
# If it's numeric, convert it to categories as shown above
# Example: data['performance rating'] = pd.cut(data['performance rating'], bins=[0, 50, 75, 100], labels=['Low', 'Medium', 'High'])

# Assume 'performance rating' is the target variable
X = data.drop('PerformanceRating', axis=1)  # Features  
y = data['PerformanceRating']                 # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier with regularization
clf = DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_split=2, min_samples_leaf=1)  # Adjust parameters as needed
clf.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(clf, X, y, cv=3)  # 5-fold cross-validation
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean CV Score: {np.mean(cv_scores):.2f}')

# Make Predictions
y_pred = clf.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=np.unique(y).astype(str))  # Ensure class names are strings
plt.title('Decision Tree Visualization')
plt.show()