import numpy as np

class Perceptron:
    def __init__(self, input_dim, learning_rate=1.0):
        self.weights = np.zeros(input_dim)
        self.bias = 0
        self.learning_rate = learning_rate
    
    def predict(self, x):
        activation = np.dot(self.weights, x) + self.bias
        return 1 if activation >= 0 else -1

    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            
            for i in range(len(X)):

                x_i = X[i]
                y_i = y[i]
                prediction = self.predict(x_i)
                if prediction != y_i:
                    self.weights += self.learning_rate*y_i*x_i
                    self.bias += self.learning_rate*y_i
            
            print(f"Epoch {epoch+1}: Weights = {self.weights}, Bias = {self.bias}")


# Sample training data (linearly separable)
X = np.array([
    [2, 3],
    [1, 1],
    [-2, -3],
    [-1, -1]
])

y = np.array([1, 1, -1, -1])

# Train the perceptron
p = Perceptron(input_dim=2)
p.train(X, y, epochs=10)

# Test prediction
print("Prediction for [3, 4]:", p.predict(np.array([3, 4])))  # Expected: 1
print("Prediction for [-2, -1]:", p.predict(np.array([-2, -1])))  # Expected: -1
