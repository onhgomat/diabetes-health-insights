# diabetes_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset (make sure 'diabetes.csv' is in the same folder)
df = pd.read_csv("diabetes.csv")

# Step 2: View first few rows
print("First 5 rows of data:")
print(df.head())

# Step 3: Basic info
print("\nData Info:")
print(df.info())

# Step 4: Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Step 5: Check missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Step 6: Correlation matrix
print("\nCorrelation Matrix:")
print(df.corr())

# Step 7: Visualization
plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.hist(df['Glucose'], bins=20, color='skyblue', edgecolor='black')
plt.title('Glucose Distribution')
plt.xlabel('Glucose Level')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.scatter(df[df['Outcome']==0]['Age'], df[df['Outcome']==0]['Glucose'], color='green', label='No Diabetes')
plt.scatter(df[df['Outcome']==1]['Age'], df[df['Outcome']==1]['Glucose'], color='red', label='Diabetes')
plt.title('Age vs Glucose')
plt.xlabel('Age')
plt.ylabel('Glucose')
plt.legend()

plt.tight_layout()
plt.show()

# Step 8: NumPy Analysis
glucose_mean = np.mean(df['Glucose'])
glucose_std = np.std(df['Glucose'])
print(f"\nGlucose Mean: {glucose_mean:.2f}, Standard Deviation: {glucose_std:.2f}")

# Step 9: Machine Learning - Logistic Regression Model
print("\n--- Machine Learning Model ---")

# Split data
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 10: Insight
if accuracy > 0.75:
    print("\n✅ Great! The model performs well for predicting diabetes risk.")
else:
    print("\n⚠️ The model accuracy is moderate — try feature scaling or tuning parameters.")

