import pandas as pd

df = pd.read_csv("loan_data.csv")

data = df.values.tolist()

attributes = [row[:-1] for row in data]
labels = [row[-1] for row in data]

hypothesis = None

for i in range(len(data)):
    if labels[i] == 'Yes':
        hypothesis = attributes[i].copy()
        break

for i in range(len(data)):
    if labels[i] == 'Yes':
        for j in range(len(hypothesis)):
            if hypothesis[j] != attributes[i][j]:
                hypothesis[j] = '?'

print(hypothesis)
