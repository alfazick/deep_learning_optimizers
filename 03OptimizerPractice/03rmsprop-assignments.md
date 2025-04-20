# RMSprop Practice Assignments

This document contains practice assignments for RMSprop, organized by difficulty level (Basic, Medium, Advanced).

## Basic Assignment
**Task**: Implement RMSprop to find the minimum of a simple function with different curvatures in different dimensions.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a function with different curvatures in different dimensions
# f(x,y) = 10*x^2 + y^2
def f(point):
    x, y = point
    return 10 * x**2 + y**2

def df(point):
    x, y = point
    return np.array([20 * x, 2 * y])

# TODO: Implement RMSprop
def rmsprop(f, df, initial_point, learning_rate, beta, epsilon, num_iterations):
    """
    Implement RMSprop.
    
    Parameters:
    - f: Function to minimize
    - df: Gradient function of f
    - initial_point: Starting point (numpy array)
    - learning_rate: Learning rate alpha
    - beta: Decay rate for squared gradients average
    - epsilon: Small constant to avoid division by zero
    - num_iterations: Number of iterations to run
    
    Returns:
    - List of points during optimization
    - List of function values during optimization
    """
    point = initial_point.copy()
    squared_grad_avg = np.zeros_like(initial_point)
    points = [point.copy()]
    values = [f(point)]
    
    # TODO: Implement RMSprop algorithm
    # For each iteration:
    # 1. Calculate gradient
    # 2. Update squared gradient average
    # 3. Update point using normalized gradient
    # 4. Append new point and function value to their respective lists
    
    return points, values

# Compare vanilla GD and RMSprop
initial_point = np.array([1.0, 1.0])
learning_rate = 0.1
beta = 0.9
epsilon = 1e-8
num_iterations = 50

# TODO: Run both vanilla GD and RMSprop and plot the results
# Include the path taken by each optimizer
```

## Medium Assignment
**Task**: Implement RMSprop for a logistic regression model on a synthetic dataset with features of different scales.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# Generate a synthetic dataset with features of different scales
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                          n_redundant=5, random_state=42)

# Scale some features to create different curvatures
X[:, :10] *= 10

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic regression model
class LogisticRegression:
    def __init__(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    
    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def compute_loss(self, X, y):
        y_pred = self.predict_proba(X)
        return -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
    
    def compute_gradients(self, X, y):
        y_pred = self.predict_proba(X)
        dw = np.dot(X.T, (y_pred - y)) / len(y)
        db = np.mean(y_pred - y)
        return dw, db

# TODO: Implement training using RMSprop
def train_rmsprop(model, X, y, learning_rate, beta, epsilon, num_iterations):
    """
    Train a logistic regression model using RMSprop.
    
    Parameters:
    - model: LogisticRegression model
    - X: Training features
    - y: Training labels
    - learning_rate: Learning rate
    - beta: Decay rate for squared gradients average
    - epsilon: Small constant to avoid division by zero
    - num_iterations: Number of iterations
    
    Returns:
    - List of loss values during training
    """
    losses = []
    
    # Initialize squared gradient average
    v_w = np.zeros_like(model.weights)
    v_b = 0
    
    # TODO: Implement RMSprop for training
    # For each iteration:
    # 1. Compute gradients
    # 2. Update squared gradient averages
    # 3. Update weights and bias using normalized gradients
    # 4. Compute and store loss
    
    return losses

# Initialize and train two models: one with vanilla GD, one with RMSprop
# Compare their convergence and final accuracy
model_vanilla = LogisticRegression(X_train.shape[1])
model_rmsprop = LogisticRegression(X_train.shape[1])

# TODO: Train both models and plot the loss curves
# Evaluate both models on the test set
```

## Advanced Assignment
**Task**: Implement RMSprop for training an RNN on a time series prediction task.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Generate a synthetic time series dataset
def generate_time_series(n_samples):
    np.random.seed(42)
    time = np.arange(n_samples)
    # Create a seasonal pattern with noise
    series = 0.5 * np.sin(0.1 * time) + 0.2 * np.sin(0.05 * time) + np.random.normal(0, 0.1, n_samples)
    return series

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Generate data
series = generate_time_series(1000)
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

# Create sequences
seq_length = 50
X, y = create_sequences(series_scaled, seq_length)

# Split into train and test
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for RNN [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define a simple RNN model
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize a simple RNN.
        
        Parameters:
        - input_size: Size of input features
        - hidden_size: Size of hidden state
        - output_size: Size of output
        """
        # Xavier initialization for weights
        self.Wxh = np.random.randn(hidden_size, input_size) * np.sqrt(2 / (input_size + hidden_size))
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2 / (hidden_size + hidden_size))
        self.Why = np.random.randn(output_size, hidden_size) * np.sqrt(2 / (hidden_size + output_size))
        
        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
    
    def forward(self, inputs):
        """
        Forward pass through the RNN.
        
        Parameters:
        - inputs: Input sequence [time_steps, batch_size, input_size]
        
        Returns:
        - outputs: Output sequence [time_steps, batch_size, output_size]
        - hidden_states: Hidden states [time_steps+1, batch_size, hidden_size]
        """
        # TODO: Implement forward pass through the RNN
        pass
    
    def backward(self, inputs, targets, hidden_states, outputs):
        """
        Backward pass through the RNN.
        
        Parameters:
        - inputs: Input sequence
        - targets: Target values
        - hidden_states: Hidden states from forward pass
        - outputs: Outputs from forward pass
        
        Returns:
        - Gradients for all parameters
        """
        # TODO: Implement backward pass through the RNN
        # Calculate gradients using backpropagation through time (BPTT)
        pass

# TODO: Implement RMSprop for training the RNN
def train_rnn_rmsprop(model, X_train, y_train, learning_rate, beta, epsilon, num_epochs):
    """
    Train an RNN using RMSprop.
    
    Parameters:
    - model: SimpleRNN model
    - X_train: Training inputs [samples, time_steps, features]
    - y_train: Training targets [samples]
    - learning_rate: Learning rate
    - beta: Decay rate for squared gradients average
    - epsilon: Small constant to avoid division by zero
    - num_epochs: Number of epochs
    
    Returns:
    - List of loss values during training
    """
    # TODO: Implement RMSprop for training the RNN
    # Initialize squared gradient averages for all parameters
    # For each epoch:
    # 1. Loop through all samples
    # 2. Perform forward pass
    # 3. Compute loss
    # 4. Perform backward pass to get gradients
    # 5. Update squared gradient averages
    # 6. Update parameters using normalized gradients
    pass

# TODO: Initialize model and train it with RMSprop
# Compare with vanilla SGD
# Plot the learning curves and prediction results
```

## Sample Solutions

### Basic Assignment Solution

```python
def rmsprop(f, df, initial_point, learning_rate, beta, epsilon, num_iterations):
    """
    Implement RMSprop.
    
    Parameters:
    - f: Function to minimize
    - df: Gradient function of f
    - initial_point: Starting point (numpy array)
    - learning_rate: Learning rate alpha
    - beta: Decay rate for squared gradients average
    - epsilon: Small constant to avoid division by zero
    - num_iterations: Number of iterations to run
    
    Returns:
    - List of points during optimization
    - List of function values during optimization
    """
    point = initial_point.copy()
    squared_grad_avg = np.zeros_like(initial_point)
    points = [point.copy()]
    values = [f(point)]
    
    for i in range(num_iterations):
        # Calculate gradient
        gradient = df(point)
        
        # Update squared gradient average
        squared_grad_avg = beta * squared_grad_avg + (1 - beta) * np.square(gradient)
        
        # Update point using normalized gradient
        point = point - learning_rate * gradient / (np.sqrt(squared_grad_avg) + epsilon)
        
        # Append new point and function value
        points.append(point.copy())
        values.append(f(point))
    
    return points, values

def vanilla_gradient_descent(f, df, initial_point, learning_rate, num_iterations):
    """Simple implementation of vanilla gradient descent for comparison"""
    point = initial_point.copy()
    points = [point.copy()]
    values = [f(point)]
    
    for i in range(num_iterations):
        gradient = df(point)
        point = point - learning_rate * gradient
        points.append(point.copy())
        values.append(f(point))
    
    return points, values

# Compare vanilla GD and RMSprop
import numpy as np
import matplotlib.pyplot as plt

initial_point = np.array([1.0, 1.0])
learning_rate = 0.1
beta = 0.9
epsilon = 1e-8
num_iterations = 50

# Run both optimizers
vanilla_points, vanilla_values = vanilla_gradient_descent(f, df, initial_point, learning_rate, num_iterations)
rmsprop_points, rmsprop_values = rmsprop(f, df, initial_point, learning_rate, beta, epsilon, num_iterations)

# Convert points to numpy arrays for easier slicing
vanilla_points = np.array(vanilla_points)
rmsprop_points = np.array(rmsprop_points)

# Plot results
plt.figure(figsize=(12, 10))

# Plot function values
plt.subplot(2, 1, 1)
plt.plot(vanilla_values, label='Vanilla GD')
plt.plot(rmsprop_values, label='RMSprop')
plt.xlabel('Iterations')
plt.ylabel('Function Value')
plt.title('Function Value vs. Iterations')
plt.legend()
plt.yscale('log')
plt.grid(True)

# Plot optimization path
plt.subplot(2, 1, 2)
plt.plot(vanilla_points[:, 0], vanilla_points[:, 1], 'b-', label='Vanilla GD')
plt.plot(rmsprop_points[:, 0], rmsprop_points[:, 1], 'r-', label='RMSprop')
plt.scatter(0, 0, c='g', marker='*', s=100, label='Optimum')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Optimization Path')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```
