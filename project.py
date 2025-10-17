# Diabetes Prediction Project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------- Step 1: Load Dataset --------
df = pd.read_csv("diabetes.csv")
print("✅ Data Loaded Successfully!\n")
print(df.head())

# -------- Step 2: Basic Info --------
print("\n📊 Dataset Info:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# -------- Step 3: Data Summary --------
print("\n📈 Statistical Summary:")
print(df.describe())

# -------- Step 4: Correlation Heatmap --------
plt.figure(figsize=(10,6))
plt.imshow(df.corr(), cmap='coolwarm', interpolation='nearest')
plt.title("Correlation Heatmap")
plt.colorbar()
plt.show()

# -------- Step 5: Data Preprocessing --------
X = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------- Step 6: Split Data --------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------- Step 7: Train Model --------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------- Step 8: Predictions --------
y_pred = model.predict(X_test)

# -------- Step 9: Evaluation --------
print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------- Step 10: Visualization --------
plt.figure(figsize=(8,5))
plt.bar(df.columns[:-1], model.coef_[0], color='teal')
plt.title("Feature Importance (Model Coefficients)")
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# -------- Step 11: Simple User Prediction --------
print("\n--- Test Your Own Data ---")
input_data = np.array([
    [6, 148, 72, 35, 0, 33.6, 0.627, 50]  # Example input
])
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)
print("Result: ", "Diabetic 🩸" if prediction[0] == 1 else "Non-Diabetic 💪")
