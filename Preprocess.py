import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('employee_dataset.csv')

# Display the initial dataset
print("Initial Dataset: ")
print(df.head(10))

# Check for missing values
print("\nMissing values: ")
print(df.isnull().sum())

# Dropping rows whose names, age, or departments are not present
df.dropna(subset=['Name', 'Age', 'Department'], inplace=True)

# Handling missing values for salaries
# Replace with mean or average salary
df['Salary'].fillna(df['Salary'].mean(), inplace=True)

# Handling NaN values in PerformanceRating with a default value
df['PerformanceRating'].fillna(3, inplace=True)

# Drop rows where 'Age' is greater than or equal to 55
df = df[df['Age'] < 55]

# Check for missing values after cleaning
print("\nMissing Values After Dropping Names, Age, and Department:")
print(df.isnull().sum())

# Convert 'Department' to dummy variables
df = pd.get_dummies(df, columns=['Department'], drop_first=True)

# Standardize numerical features (optional, depending on the algorithm)
# scaler = StandardScaler()
# df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

# Drop unnecessary columns
# Drop 'EmpID' as it is not needed for analysis
if 'EmpID' in df.columns:
    df.drop(columns=['EmpID'], inplace=True)

if 'Name' in df.columns:
    df.drop(columns=['Name'], inplace=True)

# Display the cleaned dataset
print("\nCleaned Dataset:")
print(df.head())

# Save the cleaned dataset to a new CSV file
df.to_csv('cleaned_employee_data.csv', index=False)

# Optional: Display the final DataFrame
print("\nFinal Cleaned DataFrame:")
print(df)