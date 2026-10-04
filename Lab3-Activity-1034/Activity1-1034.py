#Activity 1 (with changed Datasets)
#23- ntu-cs-1034
#Hajra Ahmad
#Topic: Student Exam Result Prediction
#Goal:wheter student will pass or fail(based studyhours,attendence etc)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)
#Load dataset
data = pd.read_csv("DataSets\student.csv")
print("Original Dataset:")
print(data)
data = pd.get_dummies(
    data,
    columns=["InternetAccess"],
    drop_first=True
)
print("\nDataset after Encoding:")
print(data)
X = data.drop("Pass", axis=1)
y = data["Pass"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)
#Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
#Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
#Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
print("\nModel Evaluation:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fail", "Pass"]
)
disp.plot()
plt.title("Confusion Matrix")
plt.show()
#ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
print("\nROC-AUC Score:", auc_score)
plt.plot(
    fpr,
    tpr,
    label="Logistic Regression (AUC = {:.2f})".format(auc_score)
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()