
import pandas as pd

df = pd.read_csv("loan_data.csv")

data = df.values.tolist()

attributes = [row[:-1] for row in data]
labels = [row[-1] for row in data]

S = attributes[0].copy()
G = [['?' for _ in range(len(S))]]

for i in range(len(data)):
    if labels[i] == 'Yes':
        for j in range(len(S)):
            if S[j] != attributes[i][j]:
                S[j] = '?'
    else:
        G_temp = []
        for g in G:
            for j in range(len(g)):
                if g[j] == '?':
                    new_g = g.copy()
                    new_g[j] = S[j]
                    if new_g[j] != attributes[i][j]:
                        G_temp.append(new_g)
        G = G_temp

print("S:", S)
print("G:", G)
