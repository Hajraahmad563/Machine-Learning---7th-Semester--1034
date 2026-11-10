import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("Shape:", df.shape)
print(df.head(10))
print("\nMissing values:\n", df.isnull().sum())
print("\nChurn distribution:\n", df["Churn"].value_counts())

# TotalCharges contains blank strings. Convert them to NaN, then use median.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

y = df["Churn"].map({"No": 0, "Yes": 1})
X = df.drop(columns=["customerID", "Churn"])
X = pd.get_dummies(X, drop_first=True, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost Default": XGBClassifier(random_state=42, eval_metric="logloss"),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, "test accuracy:", accuracy_score(y_test, model.predict(X_test)))

param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0],
}
grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="logloss"),
    param_grid=param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)
tuned_xgb = grid.best_estimator_
print("\nBest parameters:", grid.best_params_)
print("Best CV F1-score:", grid.best_score_)

for name, model in [
    ("XGBoost Default", models["XGBoost Default"]),
    ("XGBoost Tuned", tuned_xgb),
]:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred, zero_division=0))
    print("Recall:", recall_score(y_test, pred, zero_division=0))
    print("F1-score:", f1_score(y_test, pred, zero_division=0))
    print("ROC-AUC:", roc_auc_score(y_test, proba))

importance = pd.Series(tuned_xgb.feature_importances_, index=X_train.columns)
top5 = importance.sort_values(ascending=False).head(5)
print("\nTop 5 features:\n", top5)
top5.sort_values().plot(kind="barh", title="Five Most Important Features")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
