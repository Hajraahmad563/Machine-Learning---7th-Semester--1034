#activity1 
#Hajra Ahmad
#1034
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# -----------------------------------------
# Load Dataset
df = pd.read_csv("Data Set\Medical Cost Personal Datasets.csv")
print("First 5 rows:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
# -----------------------------------------
# Encode Categorical Features
le = LabelEncoder()
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])
print("\nAfter Encoding:")
print(df.head())
# -----------------------------------------
# Features and Target
X = df[['age', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']
# -----------------------------------------
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# -----------------------------------------
# Scale Numerical Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# -----------------------------------------
# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
# -----------------------------------------
# Prediction
y_pred = model.predict(X_test)
# -----------------------------------------
# Evaluation

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("\nRMSE:", rmse)
print("R2 Score:", r2)
# -----------------------------------------
# Predicted vs Actual Plot
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Insurance Cost")
plt.ylabel("Predicted Insurance Cost")
plt.title("Predicted vs Actual Insurance Costs")

plt.show()