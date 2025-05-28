from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Perceptron
# from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load only two classes for binary classification
iris = load_iris()

# Filter the dataset to include only two classes (0 and 1)
X = iris.data[iris.target != 2][:, :2]
y = iris.target[iris.target != 2]

print(f"Dataset shape: {X.shape}, Labels: {np.unique(y)}")

# 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Convert 0 labels to -1
y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

# Implement the Perceptron
class Perceptron:
    def __init__(self, learning_rate=1.0, max_iter=100):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.history = []  # Store (w, b)

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
                    self.history.append((self.w.copy(), self.b))  # Record state
                    errors += 1

             # Stop if no errors in the entire epoch
            if errors == 0:
                print(f"Epoch {epoch+1}: No errors, stopping training.")
                break
    
# Initialize and train the Perceptron
perceptron = Perceptron(learning_rate=0.1, max_iter=10)
perceptron.fit(X_train, y_train)

# Make predictions
y_pred = perceptron.predict(X_test)

# Evaluate
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Assuming X, y and perceptron are already defined and trained
fig, ax = plt.subplots()

def plot_data():
    ax.clear()
    for label in np.unique(y_train):
        ax.scatter(X_train[y_train == label][:, 0], X_train[y_train == label][:, 1],
                   label=f"Class {label}", edgecolor='k')
        
    # Dynamically set the limits with padding
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.grid(True)

def plot_decision_boundary(w, b):
    x_vals = np.array(ax.get_xlim())
    if w[1] != 0:
        y_vals = -(w[0] * x_vals + b) / w[1]
        ax.plot(x_vals, y_vals, 'k--')
    else:
        x = -b / w[0] if w[0] != 0 else 0
        ax.axvline(x=x, color='k', linestyle='--')

def animate(i):
    plot_data()
    w, b = perceptron.history[i]
    plot_decision_boundary(w, b)
    ax.set_title(f"Perceptron Update {i+1}")

# Run animation
ani = FuncAnimation(fig, animate, frames=len(perceptron.history), interval=1000, repeat=False)
plt.show()
