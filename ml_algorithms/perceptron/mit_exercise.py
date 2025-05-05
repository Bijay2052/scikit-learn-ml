import numpy as np
import matplotlib.pyplot as plt

# Sample training data (linearly separable), considering algorithm through origin
X = np.array([
    [1, -1],     # x1
    [0, 1],      # x2
    [-1.5, -1]   # x3
])
y = np.array([1, -1, 1])

# Initialize weights and bias
theta = np.array([0.0, 0.0])
epochs = 10

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
    made_mistake = False
    for i in range(len(X)):
        x_i = X[i]
        y_i = y[i]
        prediction = np.sign(np.dot(theta, x_i))
        print(f"  Prediction for x[{i+1}] = {x_i}: {prediction}, Actual y = {y_i}")
        if prediction == 0:
            prediction = -1  # treat zero as a mistake (can be adjusted based on convention)
        if prediction != y_i:
            print(f"  Mistake on x[{i+1}] = {x_i}, y = {y_i}")
            theta += y_i * x_i
            made_mistake = True
            print(f"  Updated theta: {theta}")
    if not made_mistake:
        print("No mistakes made in this epoch. Stopping early.")
        break
