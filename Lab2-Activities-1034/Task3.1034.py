
# Activity 3
# hajra ahmad
# 23-Ntu-cs-1034

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score


# -----------------------------------------
# Load Dataset
# -----------------------------------------

df = pd.read_csv("Data Set\Car Price Prediction.csv")

print("Original Dataset:")
print(df.head())

print("\nColumns:")
print(df.columns)


# -----------------------------------------
# Select Features
# -----------------------------------------

features = ['horsepower', 'citympg', 'highwaympg']

# Convert columns to numeric
for column in features:
    df[column] = pd.to_numeric(df[column], errors='coerce')

df['price'] = pd.to_numeric(df['price'], errors='coerce')

X = df[features]
y = df['price']


# -----------------------------------------
# Handle Missing Values
# -----------------------------------------

X = X.fillna(X.median())
y = y.fillna(y.median())


# -----------------------------------------
# Split Dataset
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------------------
# Linear Regression
# -----------------------------------------

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_mse = mean_squared_error(y_test, linear_pred)

linear_r2 = r2_score(y_test, linear_pred)

print("\nLinear Regression Results:")

print("MSE:", linear_mse)

print("R2 Score:", linear_r2)


# -----------------------------------------
# Polynomial Regression
# -----------------------------------------

degrees = [2, 3, 4]

results = []


for degree in degrees:

    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    mse = mean_squared_error(y_test, prediction)

    r2 = r2_score(y_test, prediction)

    results.append([degree, mse, r2])

    print("\nPolynomial Regression Degree", degree)

    print("MSE:", mse)

    print("R2 Score:", r2)


# -----------------------------------------
# Compare Results
# -----------------------------------------

results_df = pd.DataFrame(
    results,
    columns=['Degree', 'MSE', 'R2 Score']
)

print("\nPolynomial Regression Comparison:")

print(results_df)


# -----------------------------------------
# Fitted Curves
# -----------------------------------------

# Use horsepower for visualization
X_curve = df[['horsepower']]

y_curve = df['price']


# Sort values for smooth curves
x_values = pd.DataFrame({
    'horsepower': sorted(X_curve['horsepower'])
})


# -----------------------------------------
# Plot Actual Data
# -----------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    X_curve,
    y_curve,
    alpha=0.5,
    label='Actual Data'
)


# -----------------------------------------
# Linear Fitted Line
# -----------------------------------------

linear_curve = LinearRegression()

linear_curve.fit(X_curve, y_curve)

y_linear = linear_curve.predict(x_values)

plt.plot(
    x_values,
    y_linear,
    label='Linear Regression'
)


# -----------------------------------------
# Polynomial Curves
# -----------------------------------------

for degree in degrees:

    poly_model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )

    poly_model.fit(X_curve, y_curve)

    y_poly = poly_model.predict(x_values)

    plt.plot(
        x_values,
        y_poly,
        label='Polynomial Degree ' + str(degree)
    )


# -----------------------------------------
# Graph Labels
# -----------------------------------------

plt.xlabel("Horsepower")

plt.ylabel("Car Price")

plt.title("Car Price Prediction - Fitted Curves")

plt.legend()

plt.grid(True)

plt.show()