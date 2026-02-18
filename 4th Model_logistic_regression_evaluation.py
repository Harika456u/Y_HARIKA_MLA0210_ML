import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv("loan_data.csv")

le = LabelEncoder()
for column in df.columns:
    df[column] = le.fit_transform(df[column])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = LogisticRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

print("Accuracy:")
print(accuracy_score(y, y_pred))
