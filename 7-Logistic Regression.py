# Logistic Regression Algorithm
# EnjoySport Dataset

import numpy as np
from sklearn.linear_model import LogisticRegression

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

# ---------- Logistic Regression Model ----------
model = LogisticRegression()

# ---------- Training ----------
model.fit(X, y)

# ---------- Prediction ----------
predictions = model.predict(X)

print("Predicted Outputs:")
print(predictions)

# ---------- Testing with New Sample ----------
# sunny, warm, normal, strong, warm, same
test_sample = np.array([[1, 1, 1, 1, 1, 1]])
test_prediction = model.predict(test_sample)

print("Prediction for test sample:", test_prediction[0])
