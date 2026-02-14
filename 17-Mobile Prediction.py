import pandas as pd
from sklearn.linear_model import LogisticRegression

data = {
    'RAM': [2, 4, 6, 8, 12],
    'Storage': [32, 64, 128, 128, 256],
    'PriceRange': [0, 1, 1, 2, 2]   # 0=Low,1=Medium,2=High
}

df = pd.DataFrame(data)

X = df[['RAM', 'Storage']]
y = df['PriceRange']

model = LogisticRegression()
model.fit(X, y)

print("Predicted Price Range:", model.predict(X))
