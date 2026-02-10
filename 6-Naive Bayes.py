# Naive Bayes Algorithm
# EnjoySport Dataset

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# ---------- Dataset (Encoded) ----------
# sky, airtemp, humidity, wind, water, forecast
X = np.array([
    [1, 1, 1, 1, 1, 1],   # yes
    [1, 1, 0, 1, 1, 1],   # yes
    [0, 0, 0, 1, 1, 0],   # no
    [1, 1, 0, 1, 0, 0]    # yes
])

# Output
y = np.array([1, 1, 0, 1])

# ---------- Naive Bayes Model ----------
model = GaussianNB()

# ---------- Training ----------
model.fit(X, y)

# ---------- Prediction ----------
predictions = model.predict(X)

# ---------- Evaluation ----------
cm = confusion_matrix(y, predictions)
accuracy = accuracy_score(y, predictions)

print("Predicted Outputs:")
print(predictions)

print("\nConfusion Matrix:")
print(cm)

print("\nAccuracy:")
print(accuracy)
