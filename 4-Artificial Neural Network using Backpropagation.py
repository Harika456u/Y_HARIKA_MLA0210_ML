# Artificial Neural Network using Backpropagation
# EnjoySport Dataset

import numpy as np

# ---------- Activation Functions ----------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ---------- Dataset (Encoded) ----------
# sky, airtemp, humidity, wind, water, forecast
X = np.array([
    [1, 1, 1, 1, 1, 1],   # yes
    [1, 1, 0, 1, 1, 1],   # yes
    [0, 0, 0, 1, 1, 0],   # no
    [1, 1, 0, 1, 0, 0]    # yes
])

# Output
y = np.array([[1],
              [1],
              [0],
              [1]])

# ---------- Network Architecture ----------
input_neurons = 6
hidden_neurons = 4
output_neurons = 1

# ---------- Initialize Weights ----------
np.random.seed(1)
wh = np.random.uniform(size=(input_neurons, hidden_neurons))
bh = np.random.uniform(size=(1, hidden_neurons))
wo = np.random.uniform(size=(hidden_neurons, output_neurons))
bo = np.random.uniform(size=(1, output_neurons))

learning_rate = 0.1
epochs = 5000

# ---------- Training using Backpropagation ----------
for _ in range(epochs):

    # Forward Propagation
    hidden_input = np.dot(X, wh) + bh
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, wo) + bo
    output = sigmoid(final_input)

    # Error Calculation
    error = y - output

    # Backpropagation
    d_output = error * sigmoid_derivative(output)
    d_hidden = d_output.dot(wo.T) * sigmoid_derivative(hidden_output)

    # Weight Updates
    wo += hidden_output.T.dot(d_output) * learning_rate
    bo += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    wh += X.T.dot(d_hidden) * learning_rate
    bh += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# ---------- Output ----------
print("Predicted Output after Training:")
print(output.round())

# ---------- Testing with New Sample ----------
# sunny, warm, normal, strong, warm, same
test_sample = np.array([[1, 1, 1, 1, 1, 1]])

hidden_test = sigmoid(np.dot(test_sample, wh) + bh)
final_test = sigmoid(np.dot(hidden_test, wo) + bo)

print("Prediction for test sample:", round(final_test[0][0]))
