# Credit Score Classification

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample Dataset
data = {
    'Income': [25000, 50000, 75000, 100000, 40000, 60000, 85000],
    'LoanAmount': [200000, 150000, 100000, 50000, 180000, 120000, 70000],
    'CreditScore': [0, 1, 2, 2, 0, 1, 2]  
    # 0 = Low, 1 = Medium, 2 = High
}

df = pd.DataFrame(data)

X = df[['Income', 'LoanAmount']]
y = df['CreditScore']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

print("Predicted Credit Scores:", pred)
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
