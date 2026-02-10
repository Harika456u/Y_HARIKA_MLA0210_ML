# Linear Regression vs Polynomial Regression
# EnjoySport Dataset

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ---------- Dataset (Encoded) ----------
# sky, airtemp, humidity, wind, water, forecast
X = np.array([
    [1, 1, 1, 1, 1, 1],   # yes
    [1, 1, 0, 1, 1, 1],   # yes
    [0, 0, 0, 1, 1, 0],   # no
    [1, 1, 0, 1, 0, 0]    # yes
])

# Output
y = np.array([1, 1, 0, 1])

# ---------- Linear Regression ----------
linear_model = LinearRegression()
linear_model.fit(X, y)
linear_pred = linear_model.predict(X)

# ---------- Polynomial Regression ----------
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
poly_pred = poly_model.predict(X_poly)

# ---------- Output ----------
print("Linear Regression Predictions:")
print(linear_pred.round())

print("\nPolynomial Regression Predictions:")
print(poly_pred.round())
