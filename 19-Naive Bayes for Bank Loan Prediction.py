import pandas as pd
from sklearn.naive_bayes import GaussianNB

data = {
    'Income': [50000, 60000, 25000, 80000, 30000],
    'CreditScore': [700, 750, 600, 800, 580],
    'LoanApproved': [1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[['Income', 'CreditScore']]
y = df['LoanApproved']

model = GaussianNB()
model.fit(X, y)

print("Loan Prediction:", model.predict(X))
