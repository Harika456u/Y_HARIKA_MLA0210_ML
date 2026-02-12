# Car Price Prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample Dataset
data = {
    'EngineSize': [1.3, 1.5, 2.0, 2.2, 1.8],
    'Mileage': [20, 18, 15, 12, 16],
    'Price': [500000, 600000, 900000, 1200000, 800000]
}

df = pd.DataFrame(data)

X = df[['EngineSize', 'Mileage']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Predicted Price:", pred)
print("MSE:", mean_squared_error(y_test, pred))
