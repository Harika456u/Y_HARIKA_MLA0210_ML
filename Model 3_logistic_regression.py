import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("loan_data.csv")

le = LabelEncoder()
for column in df.columns:
    df[column] = le.fit_transform(df[column])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = LogisticRegression()
model.fit(X, y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Predictions:", model.predict(X))
print("Probabilities:")
print(model.predict_proba(X))
