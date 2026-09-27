# Activity 
# hajra ahmad
# 23-Ntu-cs-1034
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
# -----------------------------------------
# Load Dataset
df = pd.read_csv("Data Set\Telco Customer Churn.csv")
print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())
# -----------------------------------------
# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'], errors='coerce'
)
df = df.dropna()
# -----------------------------------------
# Encode Categorical Features
le = LabelEncoder()
df['Contract'] = le.fit_transform(df['Contract'])
df['InternetService'] = le.fit_transform(df['InternetService'])
df['Churn'] = le.fit_transform(df['Churn'])
# -----------------------------------------
# Features and Target
X = df[['tenure',
        'MonthlyCharges',
        'Contract',
        'InternetService']]
y = df['Churn']
# -----------------------------------------
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# -----------------------------------------
# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# -----------------------------------------
# Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
# -----------------------------------------
# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
# -----------------------------------------
# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("\nModel Evaluation:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
# -----------------------------------------
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()
# -----------------------------------------
# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
print("ROC AUC:", roc_auc)