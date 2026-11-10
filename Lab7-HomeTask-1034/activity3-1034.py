import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("insurance.csv")
print("Shape:", df.shape)
print(df.head(10))
print("\nMissing values:\n", df.isnull().sum())
print("\nCharges summary:\n", df["charges"].describe())

df["sex"] = df["sex"].map({"female": 0, "male": 1})
df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
X = pd.get_dummies(df.drop("charges", axis=1), columns=["region"], drop_first=True, dtype=int)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

ridge = make_pipeline(StandardScaler(), Ridge())
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gbr = GradientBoostingRegressor(random_state=42)

base_models = [
    ("ridge", ridge),
    ("rf", rf),
    ("gbr", gbr),
]
stack = StackingRegressor(
    estimators=base_models,
    final_estimator=LinearRegression(),
    cv=5
)

all_models = {
    "Ridge": ridge,
    "Random Forest": rf,
    "Gradient Boosting": gbr,
    "Stacking": stack,
}
for model in all_models.values():
    model.fit(X_train, y_train)

rows = []
predictions = {}
for name, model in all_models.items():
    pred = model.predict(X_test)
    predictions[name] = pred
    rows.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
    })
results = pd.DataFrame(rows).sort_values("RMSE")
print("\nModel comparison:\n", results.to_string(index=False))

# Stacking final estimator coefficients correspond to base-model predictions
print("\nFinal estimator coefficients:", stack.final_estimator_.coef_)
print("Final estimator intercept:", stack.final_estimator_.intercept_)
print("Coefficient magnitudes (larger absolute value = stronger linear weight):")
print(pd.Series(
    stack.final_estimator_.coef_, index=[name for name, _ in base_models]
).sort_values(key=abs, ascending=False))

plt.figure(figsize=(7, 6))
plt.scatter(y_test, predictions["Stacking"], alpha=0.65)
lims = [
    min(y_test.min(), predictions["Stacking"].min()),
    max(y_test.max(), predictions["Stacking"].max())
]
plt.plot(lims, lims, linestyle="--")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Stacking: Actual vs Predicted Charges")
plt.tight_layout()
plt.show()
