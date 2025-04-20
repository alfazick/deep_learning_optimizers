# Adam Practice Assignments

This document contains practice assignments for the Adam optimizer, organized by difficulty level (Basic, Medium, Advanced).

## Basic Assignment
**Task**: Implement Adam to find the minimum of a challenging function with multiple local minima.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a function with multiple local minima
def f(point):
    x, y = point
    return np.sin(5 * x) * np.cos(5 * y) + x**2 + y**2

def df(point):
    x, y = point
    dx = 5 * np.cos(5 * x) * np.cos(5 * y) + 2 * x
    dy = -5 * np.sin(5 * x) * np.sin(5 * y) + 2 * y
    return np.array([dx, dy])

# TODO: Implement Adam optimizer
def adam(f, df, initial_point, learning_rate, beta1, beta2, epsilon, num_iterations):
    """
    Implement Adam optimizer.
    
    Parameters:
    - f: Function to minimize
    - df: Gradient function of f
    - initial_point: Starting point (numpy array)
    - learning_rate: Learning rate alpha
    - beta1: Decay rate for first moment
    - beta2: Decay rate for second moment
    - epsilon: Small constant to avoid division by zero
    - num_iterations: Number of iterations to run
    
    Returns:
    - List of points during optimization
    - List of function values during optimization
    """
    point = initial_point.copy()
    m = np.zeros_like(initial_point)  # First moment
    v = np.zeros_like(initial_point)  # Second moment
    points = [point.copy()]
    values = [f(point)]
    
    # TODO: Implement Adam algorithm
    # For each iteration:
    # 1. Calculate gradient
    # 2. Update first moment (momentum)
    # 3. Update second moment (RMSprop)
    # 4. Apply bias correction
    # 5. Update point
    # 6. Append new point and function value to their respective lists
    
    return points, values

# Compare Adam with vanilla GD, momentum, and RMSprop
initial_point = np.array([-0.5, 0.5])
learning_rate = 0.1
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
num_iterations = 100

# TODO: Implement the other optimizers and compare their performance
# Plot the optimization paths and the function values over iterations
```

## Medium Assignment
**Task**: Implement Adam for training a neural network on a non-linearly separable dataset, comparing with other optimizers.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a non-linearly separable dataset
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a neural network with one hidden layer
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize a simple neural network with one hidden layer.
        
        Parameters:
        - input_size: Size of input features
        - hidden_size: Size of hidden layer
        - output_size: Size of output layer
        """
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / (input_size + hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / (hidden_size + output_size))
        self.b2 = np.zeros(output_size)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -30, 30)))
    
    def forward(self, X):
        """Forward pass"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def compute_loss(self, X, y):
        """Compute binary cross-entropy loss"""
        y_pred = self.forward(X)
        return -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
    
    def compute_gradients(self, X, y):
        """Compute gradients using backpropagation"""
        m = X.shape[0]
        
        # Forward pass
        y_pred = self.forward(X)
        
        # Backpropagation
        dz2 = y_pred - y.reshape(-1, 1)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0) / m
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (1 - np.power(self.a1, 2))  # Derivative of tanh
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0) / m
        
        return dW1, db1, dW2, db2
    
    def predict(self, X):
        """Make predictions"""
        return (self.forward(X) >= 0.5).astype(int)

# TODO: Implement various optimizers
def train_vanilla_gd(model, X, y, learning_rate, num_iterations):
    # Your implementation here
    pass

def train_momentum(model, X, y, learning_rate, momentum, num_iterations):
    # Your implementation here
    pass

def train_rmsprop(model, X, y, learning_rate, beta, epsilon, num_iterations):
    # Your implementation here
    pass

def train_adam(model, X, y, learning_rate, beta1, beta2, epsilon, num_iterations):
    """
    Train a neural network using Adam optimizer.
    
    Parameters:
    - model: Neural network model
    - X: Training features
    - y: Training labels
    - learning_rate: Learning rate
    - beta1: Decay rate for first moment
    - beta2: Decay rate for second moment
    - epsilon: Small constant to avoid division by zero
    - num_iterations: Number of iterations
    
    Returns:
    - List of loss values during training
    """
    losses = []
    
    # Initialize first and second moments
    m_W1 = np.zeros_like(model.W1)
    m_b1 = np.zeros_like(model.b1)
    m_W2 = np.zeros_like(model.W2)
    m_b2 = np.zeros_like(model.b2)
    
    v_W1 = np.zeros_like(model.W1)
    v_b1 = np.zeros_like(model.b1)
    v_W2 = np.zeros_like(model.W2)
    v_b2 = np.zeros_like(model.b2)
    
    # TODO: Implement Adam algorithm for training
    # For each iteration:
    # 1. Compute gradients
    # 2. Update first moments
    # 3. Update second moments
    # 4. Apply bias correction
    # 5. Update model parameters
    # 6. Compute and store loss
    
    return losses

# TODO: Train models with different optimizers and compare their performance
# Plot loss curves and decision boundaries for each optimizer
```

## Advanced Assignment
**Task**: Implement Adam with and without bias correction to demonstrate the importance of bias correction during the early stages of training.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a challenging classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                          n_redundant=5, n_clusters_per_class=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a neural network with two hidden layers
class DeepNN:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        """
        Initialize a neural network with two hidden layers.
        
        Parameters:
        - input_size: Size of input features
        - hidden_size1: Size of first hidden layer
        - hidden_size2: Size of second hidden layer
        - output_size: Size of output layer
        """
        # Xavier initialization for weights
        self.W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2 / (input_size + hidden_size1))
        self.b1 = np.zeros(hidden_size1)
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2 / (hidden_size1 + hidden_size2))
        self.b2 = np.zeros(hidden_size2)
        self.W3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2 / (hidden_size2 + output_size))
        self.b3 = np.zeros(output_size)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -30, 30)))
    
    def forward(self, X):
        """Forward pass"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.tanh(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        return self.a3
    
    def compute_loss(self, X, y):
        """Compute binary cross-entropy loss"""
        y_pred = self.forward(X)
        return -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
    
    def compute_gradients(self, X, y):
        """Compute gradients using backpropagation"""
        m = X.shape[0]
        
        # Forward pass
        y_pred = self.forward(X)
        
        # Backpropagation
        dz3 = y_pred - y.reshape(-1, 1)
        dW3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0) / m
        
        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * (1 - np.power(self.a2, 2))  # Derivative of tanh
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0) / m
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (1 - np.power(self.a1, 2))  # Derivative of tanh
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0) / m
        
        return dW1, db1, dW2, db2, dW3, db3
    
    def predict(self, X):
        """Make predictions"""
        return (self.forward(X) >= 0.5).astype(int)

# TODO: Implement Adam with and without bias correction
def train_adam_with_bias_correction(model, X, y, learning_rate, beta1, beta2, epsilon, num_iterations):
    """
    Train a neural network using Adam optimizer with bias correction.
    
    Parameters:
    - model: Neural network model
    - X: Training features
    - y: Training labels
    - learning_rate: Learning rate
    - beta1: Decay rate for first moment
    - beta2: Decay rate for second moment
    - epsilon: Small constant to avoid division by zero
    - num_iterations: Number of iterations
    
    Returns:
    - List of loss values during training
    """
    # Your implementation here
    pass

def train_adam_without_bias_correction(model, X, y, learning_rate, beta1, beta2, epsilon, num_iterations):
    """
    Train a neural network using Adam optimizer without bias correction.
    
    Parameters:
    - model: Neural network model
    - X: Training features
    - y: Training labels
    - learning_rate: Learning rate
    - beta1: Decay rate for first moment
    - beta2: Decay rate for second moment
    - epsilon: Small constant to avoid division by zero
    - num_iterations: Number of iterations
    
    Returns:
    - List of loss values during training
    """
    # Your implementation here
    pass

# TODO: Train models with both versions of Adam and compare their performance
# Focus especially on the early iterations to demonstrate the impact of bias correction
# Plot learning curves and analyze the differences
```

## Sample Solutions

### Basic Assignment Solution

```python
def adam(f, df, initial_point, learning_rate, beta1, beta2, epsilon, num_iterations):
    """
    Implement Adam optimizer.
    
    Parameters:
    - f: Function to minimize
    - df: Gradient function of f
    - initial_point: Starting point (numpy array)
    - learning_rate: Learning rate alpha
    - beta1: Decay rate for first moment
    - beta2: Decay rate for second moment
    - epsilon: Small constant to avoid division by zero
    - num_iterations: Number of iterations to run
    
    Returns:
    - List of points during optimization
    - List of function values during optimization
    """
    point = initial_point.copy()
    m = np.zeros_like(initial_point)  # First moment
    v = np.zeros_like(initial_point)  # Second moment
    points = [point.copy()]
    values = [f(point)]
    
    for t in range(1, num_iterations + 1):
        # Calculate gradient
        gradient = df(point)
        
        # Update first moment (momentum)
        m = beta1 * m + (1 - beta1) * gradient
        
        # Update second moment (RMSprop)
        v = beta2 * v + (1 - beta2) * np.square(gradient)
        
        # Apply bias correction
        m_corrected = m / (1 - beta1**t)
        v_corrected = v / (1 - beta2**t)
        
        # Update point
        point = point - learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)
        
        # Append new point and function value
        points.append(point.copy())
        values.append(f(point))
    
    return points, values

# Implementation of other optimizers for comparison
def vanilla_gd(f, df, initial_point, learning_rate, num_iterations):
    point = initial_point.copy()
    points = [point.copy()]
    values = [f(point)]
    
    for i in range(num_iterations):
        gradient = df(point)
        point = point - learning_rate * gradient
        points.append(point.copy())
        values.append(f(point))
    
    return points, values

def momentum(f, df, initial_point, learning_rate, beta, num_iterations):
    point = initial_point.copy()
    velocity = np.zeros_like(initial_point)
    points = [point.copy()]
    values = [f(point)]
    
    for i in range(num_iterations):
        gradient = df(point)
        velocity = beta * velocity + gradient
        point = point - learning_rate * velocity
        points.append(point.copy())
        values.append(f(point))
    
    return points, values

def rmsprop(f, df, initial_point, learning_rate, beta, epsilon, num_iterations):
    point = initial_point.copy()
    squared_grad_avg = np.zeros_like(initial_point)
    points = [point.copy()]
    values = [f(point)]
    
    for i in range(num_iterations):
        gradient = df(point)
        squared_grad_avg = beta * squared_grad_avg + (1 - beta) * np.square(gradient)
        point = point - learning_rate * gradient / (np.sqrt(squared_grad_avg) + epsilon)
        points.append(point.copy())
        values.append(f(point))
    
    return points, values

# Compare optimizers
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

initial_point = np.array([-0.5, 0.5])
learning_rate = 0.1
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
momentum_beta = 0.9
rmsprop_beta = 0.9
num_iterations = 100

# Run all optimizers
vanilla_points, vanilla_values = vanilla_gd(f, df, initial_point, learning_rate, num_iterations)
momentum_points, momentum_values = momentum(f, df, initial_point, learning_rate, momentum_beta, num_iterations)
rmsprop_points, rmsprop_values = rmsprop(f, df, initial_point, learning_rate, rmsprop_beta, epsilon, num_iterations)
adam_points, adam_values = adam(f, df, initial_point, learning_rate, beta1, beta2, epsilon, num_iterations)

# Convert to numpy arrays for easier manipulation
vanilla_points = np.array(vanilla_points)
momentum_points = np.array(momentum_points)
rmsprop_points = np.array(rmsprop_points)
adam_points = np.array(adam_points)

# Create a grid for contour plot
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

# Plot results
plt.figure(figsize=(16, 12))

# Plot function values vs iterations
plt.subplot(2, 2, 1)
plt.plot(vanilla_values, label='Vanilla GD')
plt.plot(momentum_values, label='Momentum')
plt.plot(rmsprop_values, label='RMSprop')
plt.plot(adam_values, label='Adam')
plt.xlabel('Iterations')
plt.ylabel('Function Value')
plt.title('Function Value vs. Iterations')
plt.legend()
plt.yscale('log')
plt.grid(True)

# Plot optimization paths on contour
plt.subplot(2, 2, 2)
plt.contour(X, Y, Z, 50, cmap='viridis')
plt.plot(vanilla_points[:, 0], vanilla_points[:, 1], 'r-', label='Vanilla GD')
plt.plot(momentum_points[:, 0], momentum_points[:, 1], 'g-', label='Momentum')
plt.plot(rmsprop_points[:, 0], rmsprop_points[:, 1], 'b-', label='RMSprop')
plt.plot(adam_points[:, 0], adam_points[:, 1], 'k-', label='Adam')
plt.scatter(vanilla_points[-1, 0], vanilla_points[-1, 1], c='r', marker='x')
plt.scatter(momentum_points[-1, 0], momentum_points[-1, 1], c='g', marker='x')
plt.scatter(rmsprop_points[-1, 0], rmsprop_points[-1, 1], c='b', marker='x')
plt.scatter(adam_points[-1, 0], adam_points[-1, 1], c='k', marker='x')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Optimization Paths')
plt.legend()
plt.grid(True)

# Plot 3D surface with paths
ax = plt.subplot(2, 1, 2, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, zorder=0)
ax.plot(vanilla_points[:, 0], vanilla_points[:, 1], [f(p) for p in vanilla_points], 'r-', label='Vanilla GD', zorder=5)
ax.plot(momentum_points[:, 0], momentum_points[:, 1], [f(p) for p in momentum_points], 'g-', label='Momentum', zorder=5)
ax.plot(rmsprop_points[:, 0], rmsprop_points[:, 1], [f(p) for p in rmsprop_points], 'b-', label='RMSprop', zorder=5)
ax.plot(adam_points[:, 0], adam_points[:, 1], [f(p) for p in adam_points], 'k-', label='Adam', zorder=5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('3D Surface with Optimization Paths')
ax.legend()

plt.tight_layout()
plt.show()
```
