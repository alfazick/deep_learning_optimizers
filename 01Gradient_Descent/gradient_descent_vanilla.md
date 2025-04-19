# Gradient Descent Visualization

A comprehensive visualization tool for understanding gradient descent optimization on quadratic functions.

## Mathematical Foundation

### The Objective Function

This application visualizes gradient descent on a general quadratic function in two variables:

$$f(w_1, w_2) = a \cdot w_1^2 + b \cdot w_2^2 + c \cdot w_1 w_2 + d \cdot w_1 + e \cdot w_2 + f$$

Where:
- $a, b$ control the curvature along the $w_1$ and $w_2$ axes
- $c$ controls the rotation of the function (cross-term)
- $d, e$ control the location of critical points (linear terms)
- $f$ is a constant offset that doesn't affect the optimization

### Gradient Calculation

The gradient of $f$ with respect to $w_1$ and $w_2$ is:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial w_1} \\ \frac{\partial f}{\partial w_2} \end{bmatrix} = \begin{bmatrix} 2a \cdot w_1 + c \cdot w_2 + d \\ 2b \cdot w_2 + c \cdot w_1 + e \end{bmatrix}$$

### Update Rule

At each iteration $t$, gradient descent updates the parameters using:

$$w_1^{(t+1)} = w_1^{(t)} - \eta \cdot \frac{\partial f}{\partial w_1}$$
$$w_2^{(t+1)} = w_2^{(t)} - \eta \cdot \frac{\partial f}{\partial w_2}$$

Where $\eta$ is the learning rate that controls the step size.

## Critical Points Analysis

### Types of Critical Points

Critical points occur where $\nabla f = 0$. For our quadratic function, we can classify them based on the eigenvalues of the Hessian matrix:

$$H = \begin{bmatrix} \frac{\partial^2 f}{\partial w_1^2} & \frac{\partial^2 f}{\partial w_1 \partial w_2} \\ \frac{\partial^2 f}{\partial w_2 \partial w_1} & \frac{\partial^2 f}{\partial w_2^2} \end{bmatrix} = \begin{bmatrix} 2a & c \\ c & 2b \end{bmatrix}$$

- If all eigenvalues are positive: Local minimum
- If all eigenvalues are negative: Local maximum
- If some eigenvalues are positive and some negative: Saddle point

### Analyzing the Presets

#### Preset 1: Convex Function (a=1, b=1, c=0, d=0, e=0, f=0)

$$f(w_1, w_2) = w_1^2 + w_2^2$$

- Gradient: $\nabla f = \begin{bmatrix} 2w_1 \\ 2w_2 \end{bmatrix}$
- Hessian: $H = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$ with eigenvalues $\lambda_1 = \lambda_2 = 2 > 0$
- This is a convex function with a global minimum at $(0,0)$
- For any starting point, gradient descent will converge to this minimum
- Geometrically, this function is a circular paraboloid (bowl shape)

#### Preset 2: Saddle Point (a=-1, b=1, c=0, d=0, e=0, f=0)

$$f(w_1, w_2) = -w_1^2 + w_2^2$$

- Gradient: $\nabla f = \begin{bmatrix} -2w_1 \\ 2w_2 \end{bmatrix}$
- Hessian: $H = \begin{bmatrix} -2 & 0 \\ 0 & 2 \end{bmatrix}$ with eigenvalues $\lambda_1 = -2 < 0$ and $\lambda_2 = 2 > 0$
- This is a saddle point at $(0,0)$
- The function increases along the $w_2$ axis (convex in this direction)
- The function decreases along the $w_1$ axis (concave in this direction)
- Geometrically, this resembles a horse saddle

**Important properties of saddle points:**
1. Saddle points are stationary points where the gradient is zero, but they are neither local minima nor maxima
2. In high-dimensional spaces (like neural networks), saddle points are more common than local minima
3. Standard gradient descent can slow down near saddle points because the gradient magnitude becomes small
4. Starting exactly on the $w_2$ axis would theoretically lead to the saddle point, but numerical instabilities typically push the solution away
5. Momentum-based optimizers help escape saddle points more efficiently

#### Preset 3: Shifted Minimum (a=1, b=1, c=0, d=-2, e=-4, f=0)

$$f(w_1, w_2) = w_1^2 + w_2^2 - 2w_1 - 4w_2$$

- Gradient: $\nabla f = \begin{bmatrix} 2w_1 - 2 \\ 2w_2 - 4 \end{bmatrix}$
- Hessian: $H = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$ with eigenvalues $\lambda_1 = \lambda_2 = 2 > 0$
- Setting the gradient to zero: $2w_1 - 2 = 0$ and $2w_2 - 4 = 0$
- This gives the minimum at $(1, 2)$
- This is still a convex function, but with the minimum shifted away from the origin
- The linear terms ($d$ and $e$) control the location of the minimum

**Properties of shifted minima:**
1. Adding linear terms to a quadratic function shifts the location of the minimum but doesn't change the shape
2. We can find the exact minimum by completing the square:
   $$f(w_1, w_2) = (w_1 - 1)^2 + (w_2 - 2)^2 - 1 - 4 = (w_1 - 1)^2 + (w_2 - 2)^2 - 5$$
3. Most machine learning loss functions have minima away from the origin, making this case particularly relevant

## Learning Rate Effects

The learning rate $\eta$ significantly impacts the optimization process:

1. **Too small**: Convergence is slow and may require many iterations
2. **Too large**: Can cause overshooting or divergence
3. **Optimal**: Balances convergence speed and stability

For a quadratic function, convergence is guaranteed when:
$$0 < \eta < \frac{2}{\lambda_{max}}$$
where $\lambda_{max}$ is the largest eigenvalue of the Hessian.

## Practical Implementation

The visualization shows:
1. 3D surface of the function
2. 2D contour plot with the gradient descent path
3. Step-by-step numerical values of the optimization process

This helps build intuition about how gradient descent navigates different types of surfaces.

## Installation & Usage

1. Save the project files in their respective locations:
   - `app.py` in the root directory
   - `index.html` in the `templates` directory
   - `script.js` in the `static` directory

2. Install dependencies:
   ```
   pip install flask numpy
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Open a browser and navigate to `http://127.0.0.1:5000`

## Connections to Machine Learning

This visualization provides insights into optimization challenges in machine learning:

1. **Loss Landscapes**: Neural network loss functions can have complex geometries with many saddle points
2. **Convergence Issues**: Understanding why training might slow down or get stuck
3. **Hyperparameter Tuning**: Visualizing the effects of learning rate on optimization
4. **Initialization**: Seeing how different starting points affect the convergence path and final solution

By experimenting with different functions and parameters, you can gain intuition about the behavior of gradient-based optimization in more complex scenarios.
