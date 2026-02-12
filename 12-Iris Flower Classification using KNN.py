# Iris Flower Classification using KNN

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# KNN Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

print("Predicted Classes:", pred)
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
