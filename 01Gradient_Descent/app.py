# app.py - Flask Backend
from flask import Flask, render_template, jsonify, request
import numpy as np
import os

app = Flask(__name__)

# Define our function and its derivatives
def objective_function(w1, w2, a=1, b=1, c=0, d=0, e=0, const=0):
    """
    General quadratic function: f(w1, w2) = aw1² + bw2² + cw1w2 + dw1 + ew2 + const
    """
    return a*w1**2 + b*w2**2 + c*w1*w2 + d*w1 + e*w2 + const

def grad_w1(w1, w2, a=1, b=1, c=0, d=0, e=0, const=0):
    """
    Partial derivative with respect to w1: ∂f/∂w1 = 2aw1 + cw2 + d
    """
    return 2*a*w1 + c*w2 + d

def grad_w2(w1, w2, a=1, b=1, c=0, d=0, e=0, const=0):
    """
    Partial derivative with respect to w2: ∂f/∂w2 = 2bw2 + cw1 + e
    """
    return 2*b*w2 + c*w1 + e

def gradient_descent(start_w1, start_w2, learning_rate, max_steps, a, b, c, d, e, const):
    """
    Perform gradient descent and return the history of steps
    """
    # Initialize
    w1 = start_w1
    w2 = start_w2
    history = []
    
    # Store initial point
    history.append({
        'step': 0,
        'w1': float(w1),
        'w2': float(w2),
        'f': float(objective_function(w1, w2, a, b, c, d, e, const)),
        'grad_w1': float(grad_w1(w1, w2, a, b, c, d, e, const)),
        'grad_w2': float(grad_w2(w1, w2, a, b, c, d, e, const))
    })
    
    # Perform gradient descent
    for step in range(1, max_steps + 1):
        # Calculate gradients
        g_w1 = grad_w1(w1, w2, a, b, c, d, e, const)
        g_w2 = grad_w2(w1, w2, a, b, c, d, e, const)
        
        # Update position
        w1 = w1 - learning_rate * g_w1
        w2 = w2 - learning_rate * g_w2
        
        # Store current point
        history.append({
            'step': step,
            'w1': float(w1),
            'w2': float(w2),
            'f': float(objective_function(w1, w2, a, b, c, d, e, const)),
            'grad_w1': float(g_w1),
            'grad_w2': float(g_w2)
        })
    
    return history

def generate_surface_data(a, b, c, d, e, const):
    """
    Generate data for 3D surface plot
    """
    # Create meshgrid
    w1_range = np.linspace(-5, 5, 30)
    w2_range = np.linspace(-5, 5, 30)
    
    surface_data = []
    
    for w1 in w1_range:
        for w2 in w2_range:
            z = objective_function(w1, w2, a, b, c, d, e, const)
            surface_data.append({
                'w1': float(w1),
                'w2': float(w2),
                'z': float(z)
            })
    
    return surface_data

def generate_contour_data(a, b, c, d, e, const):
    """
    Generate data for contour plot
    """
    # Create meshgrid
    w1_range = np.linspace(-5, 5, 50)
    w2_range = np.linspace(-5, 5, 50)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    Z = np.zeros_like(W1)
    
    # Calculate function values
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            Z[i, j] = objective_function(W1[i, j], W2[i, j], a, b, c, d, e, const)
    
    # Create levels for contour
    levels = np.linspace(np.min(Z), np.max(Z), 20)
    
    # Convert to list for JSON serialization
    w1_values = w1_range.tolist()
    w2_values = w2_range.tolist()
    z_values = Z.tolist()
    levels = levels.tolist()
    
    return {
        'w1_values': w1_values,
        'w2_values': w2_values,
        'z_values': z_values,
        'levels': levels
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_gradient_descent', methods=['POST'])
def run_gradient_descent():
    # Get parameters from request
    data = request.json
    
    a = float(data.get('a', 1))
    b = float(data.get('b', 1))
    c = float(data.get('c', 0))
    d = float(data.get('d', 0))
    e = float(data.get('e', 0))
    const = float(data.get('f', 0))  # Renamed to avoid conflicts
    
    start_w1 = float(data.get('start_w1', 2.0))
    start_w2 = float(data.get('start_w2', 2.0))
    learning_rate = float(data.get('learning_rate', 0.1))
    max_steps = int(data.get('max_steps', 20))
    
    try:
        # Run gradient descent
        history = gradient_descent(
            start_w1, start_w2, learning_rate, max_steps,
            a, b, c, d, e, const
        )
        
        # Generate surface and contour data
        surface_data = generate_surface_data(a, b, c, d, e, const)
        contour_data = generate_contour_data(a, b, c, d, e, const)
        
        return jsonify({
            'success': True,
            'history': history,
            'surface_data': surface_data,
            'contour_data': contour_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)