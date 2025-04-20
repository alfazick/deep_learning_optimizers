# AdamW Practice Assignments

This document contains practice assignments for the AdamW optimizer, organized by difficulty level (Basic, Medium, Advanced).

## Basic Assignment
**Task**: Implement AdamW and compare with Adam on a simple regularization problem.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a simple quadratic function with a regularization term
def f(point, lambda_reg=0.01):
    """
    Function with L2 regularization.
    f(x) = x^2 + lambda * ||x||^2
    """
    return np.sum(np.square(point)) + lambda_reg * np.sum(np.square(point))

def df(point, lambda_reg=0.01):
    """
    Gradient of the function with respect to x.
    df/dx = 2x + 2 * lambda * x
    """
    return 2 * point + 2 * lambda_reg * point

# TODO: Implement Adam optimizer
def adam(f, df, initial_point, learning_rate, beta1, beta2, epsilon, 
         num_iterations, lambda_reg=0.01):
    """
    Implement Adam optimizer with L2 regularization coupled to the gradient.
    
    Parameters:
    - f: Function to minimize
    - df: Gradient function of f
    - initial_point: Starting point (numpy array)
    - learning_rate: Learning rate alpha
    - beta1: Decay rate for first moment
    - beta2: Decay rate for second moment
    - epsilon: Small constant to avoid division by zero
    - num_iterations: Number of iterations to run
    - lambda_reg: L2 regularization parameter
    
    Returns:
    - List of points during optimization
    - List of function values during optimization
    """
    point = initial_point.copy()
    m = np.zeros_like(initial_point)  # First moment
    v = np.zeros_like(initial_point)  # Second moment
    points = [point.copy()]
    values = [f(point, lambda_reg)]
    
    # TODO: Implement Adam algorithm with L2 regularization coupled to gradient
    # For each iteration:
    # 1. Calculate gradient (including regularization)
    # 2. Update first moment (momentum)
    # 3. Update second moment (RMSprop)
    # 4. Apply bias correction
    # 5. Update point
    # 6. Append new point and function value to their respective lists
    
    return points, values

# TODO: Implement AdamW optimizer
def adamw(f, df, initial_point, learning_rate, beta1, beta2, epsilon, 
          num_iterations, weight_decay=0.01):
    """
    Implement AdamW optimizer with decoupled weight decay.
    
    Parameters:
    - f: Function to minimize
    - df: Gradient function of f
    - initial_point: Starting point (numpy array)
    - learning_rate: Learning rate alpha
    - beta1: Decay rate for first moment
    - beta2: Decay rate for second moment
    - epsilon: Small constant to avoid division by zero
    - num_iterations: Number of iterations to run
    - weight_decay: Weight decay parameter
    
    Returns:
    - List of points during optimization
    - List of function values during optimization
    """
    point = initial_point.copy()
    m = np.zeros_like(initial_point)  # First moment
    v = np.zeros_like(initial_point)  # Second moment
    points = [point.copy()]
    values = [f(point, 0)]  # No regularization in function, weight decay applied separately
    
    # TODO: Implement AdamW algorithm with decoupled weight decay
    # For each iteration:
    # 1. Calculate gradient (without regularization)
    # 2. Update first moment (momentum)
    # 3. Update second moment (RMSprop)
    # 4. Apply bias correction
    # 5. Update point with Adam update AND separate weight decay
    # 6. Append new point and function value to their respective lists
    
    return points, values

# Compare Adam and AdamW
initial_point = np.array([5.0, 5.0])  # Intentionally far from origin to see regularization effect
learning_rate = 0.1
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
num_iterations = 100
lambda_reg = 0.1  # Strong regularization to see the difference
weight_decay = 0.1  # Equivalent weight decay

# TODO: Run both optimizers and plot the results
# Compare the convergence paths and final weights
```

## Medium Assignment
**Task**: Implement AdamW for a linear regression model with ridge regularization, comparing with Adam on a synthetic dataset.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate a synthetic dataset
X, y, coef = make_regression(n_samples=1000, n_features=20, n_informative=10, 
                             noise=20, coef=True, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression model with ridge regularization
class LinearRegression:
    def __init__(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def compute_loss(self, X, y, lambda_reg=0):
        """Compute mean squared error with L2 regularization"""
        y_pred = self.predict(X)
        mse = np.mean((y_pred - y) ** 2)
        reg_term = lambda_reg * np.sum(self.weights ** 2)
        return mse + reg_term
    
    def compute_gradients(self, X, y, lambda_reg=0):
        """Compute gradients with L2 regularization"""
        y_pred = self.predict(X)
        m = X.shape[0]
        
        # Gradients with L2 regularization
        dw = (2/m) * np.dot(X.T, (y_pred - y)) + 2 * lambda_reg * self.weights
        db = (2/m) * np.sum(y_pred - y)
        
        return dw, db

# TODO: Implement training using Adam with coupled regularization
def train_adam(model, X, y, learning_rate, beta1, beta2, epsilon, 
               num_iterations, lambda_reg):
    """
    Train a linear regression model using Adam with coupled L2 regularization.
    
    Parameters:
    - model: LinearRegression model
    - X: Training features
    - y: Training labels
    - learning_rate: Learning rate
    - beta1: Decay rate for first moment
    - beta2: Decay rate for second moment
    - epsilon: Small constant to avoid division by zero
    - num_iterations: Number of iterations
    - lambda_reg: L2 regularization parameter
    
    Returns:
    - List of loss values during training
    """
    losses = []
    
    # Initialize moments
    m_w = np.zeros_like(model.weights)
    v_w = np.zeros_like(model.weights)
    m_b = 0
    v_b = 0
    
    # TODO: Implement Adam training with coupled L2 regularization
    # For each iteration:
    # 1. Compute gradients with L2 regularization
    # 2. Update first and second moments
    # 3. Apply bias correction
    # 4. Update model parameters
    # 5. Compute and store loss
    
    return losses

# TODO: Implement training using AdamW with decoupled weight decay
def train_adamw(model, X, y, learning_rate, beta1, beta2, epsilon, 
                num_iterations, weight_decay):
    """
    Train a linear regression model using AdamW with decoupled weight decay.
    
    Parameters:
    - model: LinearRegression model
    - X: Training features
    - y: Training labels
    - learning_rate: Learning rate
    - beta1: Decay rate for first moment
    - beta2: Decay rate for second moment
    - epsilon: Small constant to avoid division by zero
    - num_iterations: Number of iterations
    - weight_decay: Weight decay parameter
    
    Returns:
    - List of loss values during training
    """
    losses = []
    
    # Initialize moments
    m_w = np.zeros_like(model.weights)
    v_w = np.zeros_like(model.weights)
    m_b = 0
    v_b = 0
    
    # TODO: Implement AdamW training with decoupled weight decay
    # For each iteration:
    # 1. Compute gradients WITHOUT L2 regularization
    # 2. Update first and second moments
    # 3. Apply bias correction
    # 4. Update model parameters with Adam update AND separate weight decay
    # 5. Compute and store loss (without weight decay in the loss function)
    
    return losses

# Initialize models and hyperparameters
model_adam = LinearRegression(X_train.shape[1])
model_adamw = LinearRegression(X_train.shape[1])

learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
num_iterations = 500
lambda_reg = 0.1
weight_decay = 0.1

# TODO: Train both models and compare their performance
# Plot learning curves, final weights, and test set performance
# Analyze how weight decay impacts the final weights compared to L2 regularization
```

## Advanced Assignment
**Task**: Implement AdamW for training a convolutional neural network on CIFAR-10, comparing training dynamics and generalization performance with Adam.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# Define a CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# TODO: Implement custom Adam optimizer
class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        """
        Custom Adam optimizer with L2 regularization.
        
        Parameters:
        - params: Model parameters
        - lr: Learning rate
        - betas: Coefficients used for computing running averages of gradient and its square
        - eps: Term added to denominator to improve numerical stability
        - weight_decay: Weight decay (L2 penalty) applied as part of the gradient
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform a single optimization step."""
        # TODO: Implement Adam update step with L2 regularization coupled to gradients
        pass

# TODO: Implement custom AdamW optimizer
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        """
        Custom AdamW optimizer with decoupled weight decay.
        
        Parameters:
        - params: Model parameters
        - lr: Learning rate
        - betas: Coefficients used for computing running averages of gradient and its square
        - eps: Term added to denominator to improve numerical stability
        - weight_decay: Weight decay (decoupled from the gradient)
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Perform a single optimization step."""
        # TODO: Implement AdamW update step with decoupled weight decay
        pass

# Training function
def train(model, optimizer, train_loader, test_loader, epochs=5):
    """
    Train a model with the given optimizer.
    
    Parameters:
    - model: Neural network model
    - optimizer: Optimizer to use
    - train_loader: DataLoader for training data
    - test_loader: DataLoader for test data
    - epochs: Number of training epochs
    
    Returns:
    - train_losses: List of training losses
    - test_accuracies: List of test accuracies
    - weight_norms: List of L2 norms of model parameters
    """
    # TODO: Implement training loop
    # For each epoch:
    # 1. Train the model on the training set
    # 2. Evaluate on the test set
    # 3. Track training loss, test accuracy, and L2 norm of weights
    pass

# Initialize models and optimizers
model_adam = SimpleCNN()
model_adamw = SimpleCNN()

# Initialize with the same parameters
for param_adam, param_adamw in zip(model_adam.parameters(), model_adamw.parameters()):
    param_adamw.data.copy_(param_adam.data)

adam_optimizer = Adam(model_adam.parameters(), lr=0.001, weight_decay=0.01)
adamw_optimizer = AdamW(model_adamw.parameters(), lr=0.001, weight_decay=0.01)

# TODO: Train both models and analyze the results
# Compare training loss, test accuracy, and weight norms over time
# Visualize the differences in weight distributions
```

## Sample Solutions

### Basic Assignment Solution

```python
def adam(f, df, initial_point, learning_rate, beta1, beta2, epsilon, 
         num_iterations, lambda_reg=0.01):
    """
    Implement Adam optimizer with L2 regularization coupled to the gradient.
    
    Parameters:
    - f: Function to minimize
    - df: Gradient function of f
    - initial_point: Starting point (numpy array)
    - learning_rate: Learning rate alpha
    - beta1: Decay rate for first moment
    - beta2: Decay rate for second moment
    - epsilon: Small constant to avoid division by zero
    - num_iterations: Number of iterations to run
    - lambda_reg: L2 regularization parameter
    
    Returns:
    - List of points during optimization
    - List of function values during optimization
    """
    point = initial_point.copy()
    m = np.zeros_like(initial_point)  # First moment
    v = np.zeros_like(initial_point)  # Second moment
    points = [point.copy()]
    values = [f(point, lambda_reg)]
    
    for t in range(1, num_iterations + 1):
        # Calculate gradient (including regularization)
        gradient = df(point, lambda_reg)
        
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
        values.append(f(point, lambda_reg))
    
    return points, values

def adamw(f, df, initial_point, learning_rate, beta1, beta2, epsilon, 
          num_iterations, weight_decay=0.01):
    """
    Implement AdamW optimizer with decoupled weight decay.
    
    Parameters:
    - f: Function to minimize
    - df: Gradient function of f
    - initial_point: Starting point (numpy array)
    - learning_rate: Learning rate alpha
    - beta1: Decay rate for first moment
    - beta2: Decay rate for second moment
    - epsilon: Small constant to avoid division by zero
    - num_iterations: Number of iterations to run
    - weight_decay: Weight decay parameter
    
    Returns:
    - List of points during optimization
    - List of function values during optimization
    """
    point = initial_point.copy()
    m = np.zeros_like(initial_point)  # First moment
    v = np.zeros_like(initial_point)  # Second moment
    points = [point.copy()]
    values = [f(point, 0)]  # No regularization in function evaluation
    
    for t in range(1, num_iterations + 1):
        # Calculate gradient (without regularization)
        gradient = df(point, 0)  # No regularization in gradient
        
        # Update first moment (momentum)
        m = beta1 * m + (1 - beta1) * gradient
        
        # Update second moment (RMSprop)
        v = beta2 * v + (1 - beta2) * np.square(gradient)
        
        # Apply bias correction
        m_corrected = m / (1 - beta1**t)
        v_corrected = v / (1 - beta2**t)
        
        # Update point with Adam update AND separate weight decay
        adam_update = learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)
        weight_decay_update = learning_rate * weight_decay * point
        point = point - (adam_update + weight_decay_update)
        
        # Append new point and function value (without regularization for fair comparison)
        points.append(point.copy())
        values.append(f(point, 0))
    
    return points, values

# Compare Adam and AdamW
import numpy as np
import matplotlib.pyplot as plt

initial_point = np.array([5.0, 5.0])  # Intentionally far from origin to see regularization effect
learning_rate = 0.1
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
num_iterations = 100
lambda_reg = 0.1  # Strong regularization to see the difference
weight_decay = 0.1  # Equivalent weight decay

# Run both optimizers
adam_points, adam_values = adam(f, df, initial_point, learning_rate, beta1, beta2, epsilon, 
                                num_iterations, lambda_reg)
adamw_points, adamw_values = adamw(f, df, initial_point, learning_rate, beta1, beta2, epsilon, 
                                   num_iterations, weight_decay)

# Convert to numpy arrays for easier manipulation
adam_points = np.array(adam_points)
adamw_points = np.array(adamw_points)

# Plot results
plt.figure(figsize=(15, 10))

# Plot function values
plt.subplot(2, 2, 1)
plt.plot(adam_values, label='Adam')
plt.plot(adamw_values, label='AdamW')
plt.xlabel('Iterations')
plt.ylabel('Function Value')
plt.title('Function Value vs. Iterations')
plt.legend()
plt.grid(True)

# Plot optimization paths
plt.subplot(2, 2, 2)
plt.plot(adam_points[:, 0], adam_points[:, 1], 'b-', label='Adam')
plt.plot(adamw_points[:, 0], adamw_points[:, 1], 'r-', label='AdamW')
plt.scatter(0, 0, c='g', marker='*', s=100, label='Optimum')
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.title('Optimization Path')
plt.legend()
plt.grid(True)

# Plot weight magnitudes over time
plt.subplot(2, 2, 3)
adam_norms = np.linalg.norm(adam_points, axis=1)
adamw_norms = np.linalg.norm(adamw_points, axis=1)
plt.plot(adam_norms, label='Adam')
plt.plot(adamw_norms, label='AdamW')
plt.xlabel('Iterations')
plt.ylabel('L2 Norm of Weights')
plt.title('Weight Magnitude vs. Iterations')
plt.legend()
plt.grid(True)

# Plot individual weight components over time
plt.subplot(2, 2, 4)
plt.plot(adam_points[:, 0], label='Adam x[0]')
plt.plot(adam_points[:, 1], label='Adam x[1]')
plt.plot(adamw_points[:, 0], label='AdamW x[0]')
plt.plot(adamw_points[:, 1], label='AdamW x[1]')
plt.xlabel('Iterations')
plt.ylabel('Weight Value')
plt.title('Individual Weight Components vs. Iterations')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```
