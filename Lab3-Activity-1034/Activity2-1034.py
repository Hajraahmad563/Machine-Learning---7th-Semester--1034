#Activity2 (chnaged data set)
#23-ntu-cs-1034
#hajra Ahmad
#topic: House  price prediction
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
#Load dataset
df = pd.read_csv("DataSets\HousePrice.csv")
print("Original Dataset:")
print(df)
#Choose features and target
X = df[["Area_sqft", "Bedrooms", "Bathrooms", "Age"]]
y = df["Price"]
#Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)
#Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
#Linear Regression Performance
linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)
print("\nLinear Regression Performance:")
print("Mean Squared Error:", linear_mse)
print("R2 Score:", linear_r2)
#Polynomial Regression Degree 2
poly2_model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)
poly2_model.fit(X_train, y_train)
y_pred_poly2 = poly2_model.predict(X_test)
poly2_mse = mean_squared_error(y_test, y_pred_poly2)
poly2_r2 = r2_score(y_test, y_pred_poly2)
print("\nPolynomial Regression Degree 2:")
print("Mean Squared Error:", poly2_mse)
print("R2 Score:", poly2_r2)
#Polynomial Regression Degree 3
poly3_model = make_pipeline(
    PolynomialFeatures(degree=3),
    LinearRegression()
)
poly3_model.fit(X_train, y_train)
y_pred_poly3 = poly3_model.predict(X_test)
poly3_mse = mean_squared_error(y_test, y_pred_poly3)
poly3_r2 = r2_score(y_test, y_pred_poly3)
print("\nPolynomial Regression Degree 3:")
print("Mean Squared Error:", poly3_mse)
print("R2 Score:", poly3_r2)
#Polynomial Regression Degree 4
poly4_model = make_pipeline(
    PolynomialFeatures(degree=4),
    LinearRegression()
)
poly4_model.fit(X_train, y_train)
y_pred_poly4 = poly4_model.predict(X_test)
poly4_mse = mean_squared_error(y_test, y_pred_poly4)
poly4_r2 = r2_score(y_test, y_pred_poly4)
print("\nPolynomial Regression Degree 4:")
print("Mean Squared Error:", poly4_mse)
print("R2 Score:", poly4_r2)
X_area = df[["Area_sqft"]]
y_price = df["Price"]
plt.scatter(
    X_area,
    y_price,
    label="Actual Data"
)
linear_area_model = LinearRegression()
linear_area_model.fit(X_area, y_price)
area_range = pd.DataFrame({
    "Area_sqft": range(
        int(X_area.min().iloc[0]),
        int(X_area.max().iloc[0]) + 1
    )
})
linear_curve = linear_area_model.predict(area_range)
plt.plot(
    area_range,
    linear_curve,
    label="Linear Regression"
)
poly2_area_model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)
poly2_area_model.fit(X_area, y_price)
poly2_curve = poly2_area_model.predict(area_range)
plt.plot(
    area_range,
    poly2_curve,
    label="Polynomial Degree 2"
)
poly3_area_model = make_pipeline(
    PolynomialFeatures(degree=3),
    LinearRegression()
)
poly3_area_model.fit(X_area, y_price)
poly3_curve = poly3_area_model.predict(area_range)
plt.plot(
    area_range,
    poly3_curve,
    label="Polynomial Degree 3"
)
poly4_area_model = make_pipeline(
    PolynomialFeatures(degree=4),
    LinearRegression()
)
poly4_area_model.fit(X_area, y_price)
poly4_curve = poly4_area_model.predict(area_range)
plt.plot(
    area_range,
    poly4_curve,
    label="Polynomial Degree 4"
)
plt.xlabel("Area (sqft)")
plt.ylabel("House Price")
plt.title("Fitted Curves for House Price Prediction")
plt.legend()
plt.show()