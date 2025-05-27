from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Load only two classes for binary classification
iris = load_iris()

# Filter the dataset to include only two classes (0 and 1)
X = iris.data[iris.target != 2]
y = iris.target[iris.target != 2]

print(f"Dataset shape: {X.shape}, Labels: {np.unique(y)}")

# 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Implement the Perceptron
class Perceptron:
    def __init__(self, learning_rate=1.0, max_iter=100):
        self.lr = learning_rate
        self.max_iter = max_iter

    def predict(self, X):
        activation = np.dot(X, self.w) + self.b # Activation function (y_predict = sign(wx + b))
        return np.sign(activation)

    def fit(self, X, y):

        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.max_iter):
            errors = 0
            for xi, yi in zip(X, y):
                if yi * (np.dot(self.w, xi) + self.b) <= 0:
                    self.w += self.lr * yi * xi # Push weights toward the correct class
                    self.b += self.lr * yi # Push bias toward the correct class
                    errors += 1

             # Stop if no errors in the entire epoch
            if errors == 0:
                print(f"Epoch {epoch+1}: No errors, stopping training.")
                break

    # def predict(self, X):
    #     return np.where(np.dot(X, self.w) + self.b >= 0, 1, -1)
    

# Initialize and train the Perceptron
perceptron = Perceptron(learning_rate=0.1, max_iter=10)
perceptron.fit(X_train, y_train)

# Make predictions
y_pred = perceptron.predict(X_test)

# Evaluate
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy:.2f}")
