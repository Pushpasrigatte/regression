import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(42)

n = 1000
m = 3

# Generate random features
x = np.random.rand(n, m) * 10

# True weights and bias
True_w = np.array([2.0, -1.5, 0.5])
true_b = 3.0

# Generate target values with noise
y = np.dot(x, True_w) + true_b + np.random.rand(n) * 2

# Initialize weights and bias
w = np.zeros(m)
b = 0.0
alpha = 0.01
epochs = 1000

# Gradient Descent
for epoch in range(epochs):
    y_pred = np.dot(x, w) + b
    loss = (1/n) * np.sum((y - y_pred) ** 2)
    
    dw = (-2/n) * np.dot(x.T, (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    w -= alpha * dw
    b -= alpha * db
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print(f"\nOriginal Weights: {True_w}, True Bias: {true_b}")
print(f"Learned Weights: {w}, Learned Bias: {b:.4f}")

# Plotting first feature vs target
plt.scatter(x[:, 0], y, color='blue', label='Data')
plt.scatter(x[:, 0], np.dot(x, w) + b, color='red', label='Predicted')
plt.xlabel('Feature 1')
plt.ylabel('Target')
plt.legend()
plt.show()
