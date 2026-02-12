# House Price Prediction

import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Area': [1000, 1500, 1800, 2400, 3000],
    'Bedrooms': [2, 3, 3, 4, 4],
    'Price': [3000000, 5000000, 5500000, 7000000, 8500000]
}

df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

print("Predicted House Prices:", model.predict(X))
