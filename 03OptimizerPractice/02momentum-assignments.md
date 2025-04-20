# SGD with Momentum Practice Assignments

This document contains practice assignments for SGD with momentum, organized by difficulty level (Basic, Medium, Advanced).

## Basic Assignment
**Task**: Implement SGD with momentum to find the minimum of a simple quadratic function.

```python
# Implement SGD with momentum for a simple quadratic function
import numpy as np
import matplotlib.pyplot as plt

def sgd_with_momentum(f, df, initial_point, learning_rate, momentum, num_iterations):
    """
    Implement SGD with momentum.
    
    Parameters:
    - f: Function to minimize
    - df: Gradient function of f
    - initial_point: Starting point (numpy array)
    - learning_rate: Learning rate alpha
    - momentum: Momentum coefficient beta
    - num_iterations: Number of iterations to run
    
    Returns:
    - List of points during optimization
    - List of function values during optimization
    """
    point = initial_point.copy()
    velocity = np.zeros_like(initial_point)
    points = [point.copy()]
    values = [f(point)]
    
    # TODO: Implement SGD with momentum
    # Hint: For each iteration, update the velocity using the momentum formula,
    # then update the point using the velocity
    
    return points, values

# Test function: f(x) = x^2
def f(x):
    return x**2

def df(x):
    return 2*x

# Compare vanilla GD and momentum with different momentum values
initial_point = np.array([5.0])
learning_rate = 0.1
momentum_values = [0, 0.5, 0.9, 0.99]

plt.figure(figsize=(10, 6))
for momentum in momentum_values:
    points, values = sgd_with_momentum(f, df, initial_point, learning_rate, momentum, 50)
    plt.plot(values, label=f'Momentum = {momentum}')

plt.xlabel('Iterations')
plt.ylabel('Function Value')
plt.legend()
plt.title('Comparison of Different Momentum Values')
plt.yscale('log')
plt.grid(True)
```

## Medium Assignment
**Task**: Implement SGD with momentum on the Rosenbrock function and visualize how momentum helps navigate ravines.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Rosenbrock function
def rosenbrock(point):
    x, y = point
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(point):
    x, y = point
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

# TODO: Implement vanilla GD and SGD with momentum
def vanilla_gradient_descent(gradient_func, initial_point, learning_rate, num_iterations):
    # Your implementation here
    pass

def sgd_with_momentum(gradient_func, initial_point, learning_rate, momentum, num_iterations):
    # Your implementation here
    pass

# Set up the experiment
initial_point = np.array([-1.0, 1.0])
learning_rate = 0.001
momentum = 0.9
num_iterations = 1000

# Run both optimizers
vanilla_path = vanilla_gradient_descent(rosenbrock_gradient, initial_point, learning_rate, num_iterations)
momentum_path = sgd_with_momentum(rosenbrock_gradient, initial_point, learning_rate, momentum, num_iterations)

# TODO: Visualize the optimization paths on a contour plot of the Rosenbrock function
# Hint: Create a grid of points, compute the function value at each point,
# and use plt.contour() to create contour lines. Then plot the optimization paths.
```

## Advanced Assignment
**Task**: Implement mini-batch SGD with momentum for training a CNN on the MNIST dataset.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# TODO: Implement your own SGD with momentum optimizer
class SGDMomentum:
    def __init__(self, parameters, learning_rate, momentum):
        """
        Custom SGD with momentum optimizer.
        
        Parameters:
        - parameters: Model parameters
        - learning_rate: Learning rate
        - momentum: Momentum coefficient
        """
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p.data) for p in self.parameters]
    
    def step(self):
        """Update parameters using SGD with momentum"""
        # Your implementation here
        pass
    
    def zero_grad(self):
        """Zero out gradients"""
        for p in self.parameters:
            if p.grad is not None:
                p.grad.zero_()

# Initialize model and optimizer
model = SimpleCNN()
optimizer = SGDMomentum(model.parameters(), learning_rate=0.01, momentum=0.9)

# TODO: Train the model for multiple epochs
# Compare with PyTorch's built-in SGD with momentum
# Plot training loss and test accuracy for both optimizers
```

## Sample Solutions

### Basic Assignment Solution

```python
def sgd_with_momentum(f, df, initial_point, learning_rate, momentum, num_iterations):
    """
    Implement SGD with momentum.
    
    Parameters:
    - f: Function to minimize
    - df: Gradient function of f
    - initial_point: Starting point (numpy array)
    - learning_rate: Learning rate alpha
    - momentum: Momentum coefficient beta
    - num_iterations: Number of iterations to run
    
    Returns:
    - List of points during optimization
    - List of function values during optimization
    """
    point = initial_point.copy()
    velocity = np.zeros_like(initial_point)
    points = [point.copy()]
    values = [f(point)]
    
    for i in range(num_iterations):
        # Calculate gradient
        gradient = df(point)
        
        # Update velocity (momentum)
        velocity = momentum * velocity + gradient
        
        # Update point using velocity
        point = point - learning_rate * velocity
        
        # Store point and function value
        points.append(point.copy())
        values.append(f(point))
    
    return points, values

# Test function: f(x) = x^2
def f(x):
    return x**2

def df(x):
    return 2*x

# Compare vanilla GD and momentum with different momentum values
import numpy as np
import matplotlib.pyplot as plt

initial_point = np.array([5.0])
learning_rate = 0.1
momentum_values = [0, 0.5, 0.9, 0.99]

plt.figure(figsize=(10, 6))
for momentum in momentum_values:
    points, values = sgd_with_momentum(f, df, initial_point, learning_rate, momentum, 50)
    plt.plot(values, label=f'Momentum = {momentum}')

plt.xlabel('Iterations')
plt.ylabel('Function Value')
plt.legend()
plt.title('Comparison of Different Momentum Values')
plt.yscale('log')
plt.grid(True)
plt.show()
```
