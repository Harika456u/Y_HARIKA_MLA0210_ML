import pandas as pd
import math

# ---------- Dataset ----------
data = {
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast','Sunny','Sunny','Rain'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal','High','Normal','Normal'],
    'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Weak','Weak'],
    'PlayTennis': ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes']
}

df = pd.DataFrame(data)

# ---------- Entropy ----------
def entropy(target_col):
    values = target_col.unique()
    entropy_val = 0
    for val in values:
        p = target_col.value_counts()[val] / len(target_col)
        entropy_val -= p * math.log2(p)
    return entropy_val

# ---------- Information Gain ----------
def information_gain(df, attribute, target):
    total_entropy = entropy(df[target])
    values = df[attribute].unique()

    weighted_entropy = 0
    for val in values:
        subset = df[df[attribute] == val]
        weighted_entropy += (len(subset)/len(df)) * entropy(subset[target])

    return total_entropy - weighted_entropy

# ---------- ID3 Algorithm ----------
def id3(df, target, attributes):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]

    if len(attributes) == 0:
        return df[target].mode()[0]

    gains = {attr: information_gain(df, attr, target) for attr in attributes}
    best_attr = max(gains, key=gains.get)

    tree = {best_attr: {}}
    remaining_attrs = [attr for attr in attributes if attr != best_attr]

    for val in df[best_attr].unique():
        subset = df[df[best_attr] == val]
        tree[best_attr][val] = id3(subset, target, remaining_attrs)

    return tree

# ---------- Build Tree ----------
attributes = list(df.columns[:-1])
decision_tree = id3(df, 'PlayTennis', attributes)

print("Decision Tree:")
print(decision_tree)

# ---------- Classification ----------
def classify(sample, tree):
    if not isinstance(tree, dict):
        return tree

    attribute = next(iter(tree))
    value = sample.get(attribute)

    if value in tree[attribute]:
        return classify(sample, tree[attribute][value])
    else:
        return "Unknown"

new_sample = {
    'Outlook': 'Sunny',
    'Temperature': 'Cool',
    'Humidity': 'Normal',
    'Wind': 'Weak'
}

print("Prediction for new sample:", classify(new_sample, decision_tree))
