#23-NTU-CS-1034
#MACHINE LEARNING LAB 08
#SLIDES 28 TO 38
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import (
    train_test_split,StratifiedKFold, cross_val_predict,cross_val_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
raw_data = pd.read_csv("titanic.csv")

feature_columns = [
    "Pclass", "Gender", "Age", "SibSp", "Parch", "Fare"
]
data = raw_data[["Survived"] + feature_columns].copy()
# Encode Gender
encoder = LabelEncoder()
data["Gender"] = encoder.fit_transform(
    data["Gender"].astype(str)
)
# Handle missing Age values
age_imputer = SimpleImputer(strategy="median")
data[["Age"]] = age_imputer.fit_transform(
    data[["Age"]]
)
# Separate features and target
X = data.drop("Survived", axis=1)
y = data["Survived"]
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)
depth_values = [2, 4, 6, 8, None]
results = []
for depth in depth_values:
    model = RandomForestClassifier(
        max_depth=depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    train_accuracy = accuracy_score(
        y_train, model.predict(X_train)
    )
    val_accuracy = accuracy_score(
        y_val, model.predict(X_val)
    )
    results.append([
        depth,
        train_accuracy,
        val_accuracy
    ])
split_results = pd.DataFrame(
    results,
    columns=[
        "Max Depth",
        "Training Accuracy",
        "Validation Accuracy"
    ]
)
print("\nTrain and Validation Results:")
print(split_results)
best_depth = split_results.loc[
    split_results["Validation Accuracy"].idxmax(),
    "Max Depth"
]
if pd.isna(best_depth):
    best_depth = None
print("\nBest depth:", best_depth)
#SLIDES 28
folds = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
model = RandomForestClassifier(random_state=42)
predictions = cross_val_predict(
    model, X, y, cv=folds
)
probabilities = cross_val_predict(
    model,
    X,
    y,
    cv=folds,
    method="predict_proba"
)[:, 1]
threshold_results = []
for threshold in np.arange(0.1, 1.0, 0.1):
    threshold_predictions = (
        probabilities >= threshold
    ).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y, threshold_predictions
    ).ravel()
    threshold_results.append({
        "Threshold": round(threshold, 2),
        "Accuracy": accuracy_score(
            y, threshold_predictions
        ),
        "Precision": precision_score(
            y, threshold_predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y, threshold_predictions,
            zero_division=0
        ),
        "F1-score": f1_score(
            y, threshold_predictions,
            zero_division=0
        ),
        "Specificity": tn / (tn + fp)
    })
threshold_table = pd.DataFrame(threshold_results)
print("\nClassification Threshold Comparison:")
print(threshold_table)
# Find the threshold with the highest F1-score
best_row = threshold_table.loc[
    threshold_table["F1-score"].idxmax()
]
print("\nBest threshold:", best_row["Threshold"])
print(
    "Best F1-score:",
    round(best_row["F1-score"], 3)
)
depths = [
    1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, None
]
fit_results = []
folds = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
for depth in depths:
    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )
    # Training accuracy
    tree.fit(X, y)
    train_accuracy = accuracy_score(
        y, tree.predict(X)
    )
    # Cross-validation accuracy
    cv_accuracy = cross_val_score(
        tree, X, y, cv=folds
    ).mean()
    fit_results.append({
        "Max Depth": str(depth),
        "Training Accuracy": train_accuracy,
        "Cross-validation Accuracy": cv_accuracy
    })
fit_table = pd.DataFrame(fit_results)
print("\nOverfitting and Underfitting Results:")
print(fit_table)

plt.figure(figsize=(8, 5))

plt.plot(
    fit_table["Max Depth"],
    fit_table["Training Accuracy"],
    marker="o",
    label="Training"
)
plt.plot(
    fit_table["Max Depth"],
    fit_table["Cross-validation Accuracy"],
    marker="o",
    label="Cross-validation"
)
plt.xlabel("Decision Tree Maximum Depth")
plt.ylabel("Accuracy")
plt.title("Overfitting and Underfitting")
plt.legend()
plt.grid(True)
plt.show()
died = data[data["Survived"] == 0]
survivors = data[
    data["Survived"] == 1
].sample(47, random_state=42)
imbalanced_data = pd.concat(
    [died, survivors]
).sample(frac=1, random_state=42)
X_imb = imbalanced_data.drop(
    "Survived", axis=1
)
y_imb = imbalanced_data["Survived"]
print("\nClass counts:")
print(y_imb.value_counts())
print(
    "Survivor percentage:",
    round(y_imb.mean() * 100, 2),
    "%"
)
# Select one training fold
folds = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
train_index, test_index = next(
    folds.split(X_imb, y_imb)
)
X_train = X_imb.iloc[train_index]
y_train = y_imb.iloc[train_index]
print("\nBefore SMOTE:")
print(y_train.value_counts())
# Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = (
    smote.fit_resample(X_train, y_train)
)
print("\nAfter SMOTE:")
print(
    pd.Series(y_train_balanced).value_counts()
)
# Train final model using all labelled data
final_model = RandomForestClassifier(
    max_depth=best_depth,
    n_estimators=100,
    random_state=42
)
final_model.fit(X, y)
# Save model and preprocessing information
model_bundle = {
    "model": final_model,
    "gender_encoder": encoder,
    "age_imputer": age_imputer,
    "feature_columns": list(X.columns),
    "target_name": "Survived"
}
joblib.dump(
    model_bundle,
    "titanic_random_forest_model.joblib"
)
print(
    "\nFinal model saved as "
    "titanic_random_forest_model.joblib"
)
loaded_bundle = joblib.load(
    "titanic_random_forest_model.joblib"
)
loaded_model = loaded_bundle["model"]
print(
    "Loaded model prediction for the first passenger:",
    loaded_model.predict(X.iloc[[0]])
)
