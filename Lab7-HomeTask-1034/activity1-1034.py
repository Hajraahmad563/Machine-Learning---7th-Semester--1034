import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("water_potability.csv")
print("Shape:", df.shape)
print(df.head(10))
print("\nMissing values:\n", df.isnull().sum())
print("\nPotability distribution:\n", df["Potability"].value_counts())

X = df.drop("Potability", axis=1)
y = df["Potability"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Fit imputer on training data only, then transform test data.
imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)

lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
svc = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
rf = RandomForestClassifier(n_estimators=100, random_state=42)

models = [
    ("lr", lr),
    ("svc", svc),
    ("rf", rf),
]
single_models = {
    "Logistic Regression": lr,
    "SVC": svc,
    "Random Forest": rf,
}
for model in single_models.values():
    model.fit(X_train, y_train)

voting_hard = VotingClassifier(estimators=models, voting="hard")
voting_soft = VotingClassifier(estimators=models, voting="soft")
voting_hard.fit(X_train, y_train)
voting_soft.fit(X_train, y_train)

single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)

bag_10 = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42), n_estimators=10,
    max_samples=0.5, bootstrap=True, random_state=42
)
bag_100 = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42), n_estimators=100,
    max_samples=0.5, bootstrap=True, random_state=42
)
pasting_100 = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42), n_estimators=100,
    max_samples=0.5, bootstrap=False, random_state=42
)

for model in (bag_10, bag_100, pasting_100):
    model.fit(X_train, y_train)

all_models = {
    **single_models,
    "Single Decision Tree": single_tree,
    "Voting Hard": voting_hard,
    "Voting Soft": voting_soft,
    "Bagging 10 (bootstrap)": bag_10,
    "Bagging 100 (bootstrap)": bag_100,
    "Pasting 100": pasting_100,
}
rows = []
for name, model in all_models.items():
    pred = model.predict(X_test)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0),
    })

results = pd.DataFrame(rows).sort_values("F1-score", ascending=False)
print("\nModel comparison:\n", results.to_string(index=False))
best_name = results.iloc[0]["Model"]
best_model = all_models[best_name]
print("\nBest model by F1-score:", best_name)
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.title(f"Confusion Matrix: {best_name}")
plt.tight_layout()
plt.show()

single_best = results[results["Model"].isin(single_models.keys())]["F1-score"].max()
ensemble_best = results[results["Model"].isin([
    "Voting Hard", "Voting Soft", "Bagging 10 (bootstrap)",
    "Bagging 100 (bootstrap)", "Pasting 100"
])]["F1-score"].max()
print("\nDid an ensemble beat the best single model?",
      "Yes" if ensemble_best > single_best else "No")
