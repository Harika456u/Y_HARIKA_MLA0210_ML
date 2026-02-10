# Expectation Maximization Algorithm
# EnjoySport Dataset

import numpy as np
from sklearn.mixture import GaussianMixture

# ---------- Dataset (Encoded) ----------
# sky, airtemp, humidity, wind, water, forecast
X = np.array([
    [1, 1, 1, 1, 1, 1],   # yes
    [1, 1, 0, 1, 1, 1],   # yes
    [0, 0, 0, 1, 1, 0],   # no
    [1, 1, 0, 1, 0, 0]    # yes
])

# ---------- EM Model (Gaussian Mixture) ----------
# n_components = number of clusters
gmm = GaussianMixture(n_components=2, random_state=0)

# ---------- Training ----------
gmm.fit(X)

# ---------- Prediction (Clustering) ----------
labels = gmm.predict(X)

print("Cluster Labels for each data point:")
print(labels)
