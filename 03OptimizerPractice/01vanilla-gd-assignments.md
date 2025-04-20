# Vanilla Gradient Descent Practice Assignments

This document contains practice assignments for vanilla gradient descent, organized by difficulty level (Basic, Medium, Advanced).

## Basic Assignment
**Task**: Implement vanilla gradient descent to find the minimum of a simple quadratic function.

```python
# Complete the following function to implement vanilla gradient descent
def vanilla_gradient_descent(f, df, initial_point, learning_rate, num_iterations):
    """
    Implement vanilla gradient descent.
    
    Parameters:
    - f: Function to minimize
    - df: Gradient function of f
    - initial_point: Starting point (numpy array)
    - learning_rate: Learning rate alpha
    - num_iterations: Number of iterations to run
    
    Returns:
    - List of points during optimization
    - List of function values during optimization
    """
    point = initial_point.copy()
    points = [point.copy()]
    values = [f(point)]
    
    # TODO: Implement vanilla gradient descent
    # Hint: For each iteration, calculate the gradient, update the point,
    # and append the new point and function value to their respective lists
    
    return points, values

# Test function: f(x) = x^2
def f(x):
    return x**2

def df(x):
    return 2*x

# Example usage
import numpy as np
initial_point = np.array([5.0])
points, values = vanilla_gradient_descent(f, df, initial_point, 0.1, 50)
```

## Medium Assignment
**Task**: Compare the convergence of vanilla gradient descent with different learning rates on the Rosenbrock function.

```python
# Implement vanilla gradient descent for the Rosenbrock function with different learning rates
import numpy as np
import matplotlib.pyplot as plt

# Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
def rosenbrock(point):
    x, y = point
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(point):
    x, y = point
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

# TODO: Implement vanilla gradient descent
def vanilla_gradient_descent(gradient_func, initial_point, learning_rate, num_iterations):
    """
    Implement vanilla gradient descent.
    
    Parameters:
    - gradient_func: Gradient function
    - initial_point: Starting point (numpy array)
    - learning_rate: Learning rate alpha
    - num_iterations: Number of iterations to run
    
    Returns:
    - List of points during optimization
    """
    # Your implementation here
    pass

# TODO: Compare different learning rates
learning_rates = [0.0001, 0.001, 0.01]
initial_point = np.array([-1.0, 1.0])
num_iterations = 1000

# For each learning rate, run gradient descent and plot the convergence
# Hint: Plot the distance to the optimal point [1,1] versus iteration number
```

## Advanced Assignment
**Task**: Implement vanilla gradient descent for a simple neural network to classify a synthetic dataset.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# TODO: Implement a simple neural network with one hidden layer
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize the neural network with random weights"""
        # Your implementation here
        pass
    
    def forward(self, X):
        """Forward pass"""
        # Your implementation here
        pass
    
    def compute_loss(self, X, y):
        """Compute binary cross-entropy loss"""
        # Your implementation here
        pass
    
    def compute_gradients(self, X, y):
        """Compute gradients of the loss with respect to weights"""
        # Your implementation here
        pass
    
    def train_vanilla_gd(self, X, y, learning_rate, num_iterations):
        """Train the network using vanilla gradient descent"""
        # Your implementation here
        pass
    
    def predict(self, X):
        """Make predictions"""
        # Your implementation here
        pass

# Create and train the network
input_size = 2
hidden_size = 20
output_size = 1
nn = SimpleNN(input_size, hidden_size, output_size)

# Train the network and plot training loss over iterations
# Evaluate on test set and visualize decision boundary
```

## Sample Solutions

### Basic Assignment Solution

```python
def vanilla_gradient_descent(f, df, initial_point, learning_rate, num_iterations):
    """
    Implement vanilla gradient descent.
    
    Parameters:
    - f: Function to minimize
    - df: Gradient function of f
    - initial_point: Starting point (numpy array)
    - learning_rate: Learning rate alpha
    - num_iterations: Number of iterations to run
    
    Returns:
    - List of points during optimization
    - List of function values during optimization
    """
    point = initial_point.copy()
    points = [point.copy()]
    values = [f(point)]
    
    for i in range(num_iterations):
        # Calculate gradient
        gradient = df(point)
        
        # Update point
        point = point - learning_rate * gradient
        
        # Store point and function value
        points.append(point.copy())
        values.append(f(point))
    
    return points, values

# Test function: f(x) = x^2
def f(x):
    return x**2

def df(x):
    return 2*x

# Example usage
import numpy as np
import matplotlib.pyplot as plt

initial_point = np.array([5.0])
points, values = vanilla_gradient_descent(f, df, initial_point, 0.1, 50)

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(points)
plt.xlabel('Iteration')
plt.ylabel('Parameter Value')
plt.title('Parameter Values During Optimization')

plt.subplot(1, 2, 2)
plt.plot(values)
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.title('Function Values During Optimization')
plt.yscale('log')

plt.tight_layout()
plt.show()
```
