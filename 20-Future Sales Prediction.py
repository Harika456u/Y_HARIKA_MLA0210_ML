import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Month': [1, 2, 3, 4, 5, 6],
    'Sales': [200, 250, 300, 350, 400, 450]
}

df = pd.DataFrame(data)

X = df[['Month']]
y = df['Sales']

model = LinearRegression()
model.fit(X, y)

future_month = [[7]]
prediction = model.predict(future_month)

print("Predicted Sales for Month 7:", prediction[0])
