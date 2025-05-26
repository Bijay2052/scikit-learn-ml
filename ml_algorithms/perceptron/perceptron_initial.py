import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Linearly separable synthetic data (2D)
X = np.array([
    [2, 3], [4, 2], [3, 6], [4, 5],  # Class +1
    [1, 1], [2, 0], [0, 2], [1, 3]   # Class -1
])

y = np.array([1, 1, 1, 1, -1, -1, -1, -1])  # Class labels

# Perceptron implementation
class Perceptron:
    def __init__(self, learning_rate=1.0, max_iter=10):
        # Initialize weights and bias
        self.lr = learning_rate
        self.max_iter = max_iter
        self.w = None
        self.b = 0
        self.history = []

    def predict(self, x, y=None):
        if self.w is None:
            raise ValueError("Model has not been trained yet. Call 'train' method first.")
        activation = np.dot(self.weights, x) + self.bias # Activation function (y_predict = sign(wx + b))
        return np.sign(activation)

    def fit(self, X, y):

        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.max_iter):
            error_made = False
            for xi, yi in zip(X, y):
                if yi * (np.dot(self.w, xi) + self.b) <= 0:
                    self.w += self.lr * yi * xi # Push weights toward the correct class
                    self.b += self.lr * yi # Push bias toward the correct class
                    self.history.append((self.w.copy(), self.b)) # Save weights/bias for plotting the animation
                    print(f"Epoch {epoch+1}, Sample {xi}, Update: w = {self.w}, b = {self.b}, Wrongly classified")
                    error_made = True

             # Stop if no errors in the entire epoch
            if not error_made:
                print(f"Training stopped after {epoch+1} epochs.")
                break

perceptron = Perceptron()
perceptron.fit(X, y)

# Plotting function setup
fig, ax = plt.subplots()

def plot_data():
    ax.clear()
    for label in np.unique(y):
        ax.scatter(X[y == label][:, 0], X[y == label][:, 1],
                   label=f"Class {label}", edgecolor='k')
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 7)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()

def plot_decision_boundary(w, b):
    x_vals = np.array(ax.get_xlim())
    if w[1] != 0:
        y_vals = -(w[0] * x_vals + b) / w[1]
        ax.plot(x_vals, y_vals, 'k--')
    else:
        # Vertical line
        x = -b / w[0] if w[0] != 0 else 0
        ax.axvline(x=x, color='k', linestyle='--')

# Animation function
def animate(i):
    plot_data()
    w, b = perceptron.history[i]
    plot_decision_boundary(w, b)
    ax.set_title(f"Perceptron Iteration {i+1}")

anim = FuncAnimation(fig, animate, frames=len(perceptron.history), interval=1000, repeat=False)
plt.tight_layout()
plt.show()
