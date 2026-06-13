## Functions of N Variables

*Essential Mathematics for ML — Structured Notes*

---

## 1. From Single Variable to N Variables

### Motivation and Intuition
A single-variable function $f: \mathbb{R} \to \mathbb{R}$ maps a line to a line. But machine learning operates in high-dimensional spaces — a house price prediction model takes in 10+ features, an image classifier takes in $28 \times 28 = 784$ pixel values, and a language model processes thousands of tokens simultaneously.

**Functions of $n$ variables** extend the concept of a function to $n$-dimensional input spaces. They are the mathematical language of every ML model, loss function, and optimization landscape.

### Formal Definition
Let $S \subseteq \mathbb{R}^n$. A function of $n$ variables is a mapping:

$$
f: S \subseteq \mathbb{R}^n \to \mathbb{R}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $f$ | Function of $n$ variables | Maps an $n$-dimensional input to a scalar output |
| $S \subseteq \mathbb{R}^n$ | Domain (subset of $\mathbb{R}^n$) | The set of valid $n$-dimensional input vectors |
| $\mathbb{R}^n$ | $n$-dimensional Euclidean space | The space of all $n$-tuples of real numbers |
| $\to \mathbb{R}$ | Output is a real number | The function produces a single scalar value (e.g., loss, probability) |

$$
w = f(x_1, x_2, \dots, x_n)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $w$ | Dependent variable (output) | The scalar result of evaluating $f$ at the given input |
| $x_1, x_2, \dots, x_n$ | Independent variables (features) | The $n$ input coordinates; each $x_i$ is one feature dimension |
| $f(x_1, \dots, x_n)$ | Function evaluation | The mapping from features to output — e.g., $f(\text{bedrooms}, \text{sqft}) = \text{price}$ |

where $x_1, x_2, \dots, x_n$ are the **independent variables** (features) and $w$ is the **dependent variable** (output).

### What is $\mathbb{R}^n$?

$\mathbb{R}^n$ is the set of all $n$-tuples of real numbers:

$$
\mathbb{R}^n = \{(x_1, x_2, \dots, x_n) : x_i \in \mathbb{R} \; \forall i\}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbb{R}^n$ | $n$-dimensional Euclidean space | The set of all ordered $n$-tuples of real numbers |
| $(x_1, x_2, \dots, x_n)$ | A point/vector in $\mathbb{R}^n$ | Each $x_i$ is a coordinate along the $i$-th axis |
| $x_i \in \mathbb{R}$ | Each coordinate is real | Every component can be any real number |
| $\forall i$ | For all indices $i = 1, \dots, n$ | Every coordinate of the tuple is a real number |

| Space | Dimensions | Geometric Interpretation |
|:---|:---|:---|
| $\mathbb{R}^1$ | 1 | A number line |
| $\mathbb{R}^2$ | 2 | A plane $(x, y)$ |
| $\mathbb{R}^3$ | 3 | 3D space $(x, y, z)$ |
| $\mathbb{R}^n$ | $n$ | $n$-dimensional hyperplane (cannot visualize for $n > 3$) |

```python
import numpy as np

# A function of 3 variables
def f(x1, x2, x3):
    return x1**2 + x2*x3 - np.sin(x1)

# Evaluating at a point in R^3
result = f(1.0, 2.0, 3.0)  # 1 + 6 - sin(1) ≈ 6.158
```

---

## 2. Domain and Range

### Domain
The **domain** of $f(x_1, \dots, x_n)$ is the set of all points $(x_1, \dots, x_n) \in \mathbb{R}^n$ for which $f$ is defined.

### Range
The **range** is the set of all output values $f$ actually produces:

$$
\text{Range}(f) = \{f(x_1, \dots, x_n) : (x_1, \dots, x_n) \in \text{Domain}\}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\text{Range}(f)$ | Range (image) of $f$ | The set of all actual output values the function can produce |
| $\{f(x_1, \dots, x_n)\}$ | Set of all function outputs | Collects every possible $f$ value over all valid inputs |
| $(x_1, \dots, x_n) \in \text{Domain}$ | All domain points | The set comprehension ranges over the entire domain |
| Range $\subseteq$ Codomain | Range is subset of codomain | Not every codomain element is necessarily achieved |

### Example: Domain in ML

For the sigmoid function applied to a vector:

$$
f(\mathbf{x}) = \frac{1}{1 + e^{-\mathbf{x}}}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $f(\mathbf{x})$ | Sigmoid applied element-wise to vector $\mathbf{x}$ | Each component of $\mathbf{x}$ is squashed independently to $(0, 1)$ |
| $\frac{1}{1 + e^{-\mathbf{x}}}$ | Element-wise sigmoid | For each component $x_i$, output is $\frac{1}{1 + e^{-x_i}}$ |
| $\mathbf{x}$ | Input vector in $\mathbb{R}^n$ | Can be any real-valued $n$-dimensional vector (unbounded domain) |
| $\text{Domain: } \mathbb{R}^n$ | All real vectors | The sigmoid accepts any real input — no restrictions |
| $\text{Range: } (0, 1)^n$ | Each output in $(0, 1)$ | Outputs are bounded and interpretable as probabilities per component |

* **Domain:** $\mathbb{R}^n$ (any real-valued vector)
* **Range:** $(0, 1)^n$ (each component is a probability)

```python
import numpy as np

# Domain: any vector in R^n
x = np.array([100.0, -100.0, 0.0])

# Range: always (0, 1) per component
output = 1 / (1 + np.exp(-x))
print(output)  # [1.0, 0.0, 0.5] — numerically clamped
```

---

## 3. Level Sets and Surfaces

For a function $f: \mathbb{R}^n \to \mathbb{R}$, a **level set** (or contour) is the set of all input points that produce the same output:

$$
L_c = \{(x_1, \dots, x_n) : f(x_1, \dots, x_n) = c\}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $L_c$ | Level set at value $c$ | The set of all inputs that produce the same output $c$ |
| $\{(x_1, \dots, x_n)\}$ | Set of $n$-dimensional points | All input coordinates that satisfy the equation |
| $f(x_1, \dots, x_n) = c$ | Constant-output condition | The function equals exactly $c$ at every point in the level set |
| Level set | Contour of constant value | Gradient is perpendicular to level sets; GD crosses them orthogonally |

* In $\mathbb{R}^2$, level sets are **contour curves**.
* In $\mathbb{R}^3$, level sets are **contour surfaces**.

**ML Connection:** The loss landscape of a neural network is a function $\mathcal{L}: \mathbb{R}^d \to \mathbb{R}$ where $d$ is the number of parameters. Gradient descent navigates this landscape by following the direction of steepest descent, crossing level sets perpendicularly.

```python
import numpy as np
import matplotlib.pyplot as plt

# 2D loss function: f(x, y) = x^2 + y^2 (a bowl)
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

# Level sets are concentric circles
plt.contour(X, Y, Z, levels=10)
plt.title("Level Sets of f(x,y) = x^2 + y^2")
plt.xlabel("x"); plt.ylabel("y")
plt.axis("equal")
```

---

## 4. Partial Derivatives

### Definition
For $f(x_1, x_2, \dots, x_n)$, the **partial derivative** with respect to $x_i$ measures how $f$ changes when *only* $x_i$ varies, holding all other variables constant:

$$
\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \dots, x_i + h, \dots, x_n) - f(x_1, \dots, x_n)}{h}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\frac{\partial f}{\partial x_i}$ | Partial derivative w.r.t. $x_i$ | Measures how $f$ changes when ONLY $x_i$ varies, all other inputs fixed |
| $h$ | Small perturbation to $x_i$ | An infinitesimally small increment added to the $i$-th coordinate |
| $f(\dots, x_i + h, \dots) - f(\dots)$ | Change in $f$ due to $h$ | The difference in output caused by perturbing $x_i$ by $h$ |
| $\lim_{h \to 0}$ | Limit as perturbation vanishes | The instantaneous rate of change of $f$ in the $x_i$ direction |

### Geometric Interpretation
In $\mathbb{R}^3$, if $z = f(x, y)$, then $\frac{\partial f}{\partial x}$ is the slope of the surface in the $x$-direction, and $\frac{\partial f}{\partial y}$ is the slope in the $y$-direction.

### Example

$$
f(x, y) = x^2 y + \sin(xy)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $f(x, y) = x^2 y + \sin(xy)$ | Example function of two variables | A mixed polynomial-trigonometric function used to demonstrate partial derivatives |
| $x^2 y$ | Polynomial term | Product of $x^2$ and $y$ — contributes to both partial derivatives |
| $\sin(xy)$ | Trigonometric term | Sine of product $xy$ — requires chain rule for differentiation |

$$
\frac{\partial f}{\partial x} = 2xy + y\cos(xy)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\frac{\partial f}{\partial x}$ | Partial derivative w.r.t. $x$ | Differentiate $x^2y$ as $2xy$, and $\sin(xy)$ as $y\cos(xy)$ using chain rule |
| $2xy$ | Derivative of $x^2 y$ w.r.t. $x$ | Treat $y$ as constant; derivative of $x^2$ is $2x$, times $y$ |
| $y\cos(xy)$ | Derivative of $\sin(xy)$ w.r.t. $x$ | Chain rule: $\frac{d}{dx}\sin(xy) = \cos(xy) \cdot y$ |

$$
\frac{\partial f}{\partial y} = x^2 + x\cos(xy)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\frac{\partial f}{\partial y}$ | Partial derivative w.r.t. $y$ | Differentiate $x^2y$ as $x^2$, and $\sin(xy)$ as $x\cos(xy)$ |
| $x^2$ | Derivative of $x^2 y$ w.r.t. $y$ | Treat $x$ as constant; derivative of $y$ is $1$, times $x^2$ |
| $x\cos(xy)$ | Derivative of $\sin(xy)$ w.r.t. $y$ | Chain rule: $\frac{d}{dy}\sin(xy) = \cos(xy) \cdot x$ |

```python
import sympy as sp

x, y = sp.symbols('x y')
f = x**2 * y + sp.sin(x * y)

# Partial derivatives
df_dx = sp.diff(f, x)  # 2*x*y + y*cos(x*y)
df_dy = sp.diff(f, y)  # x**2 + x*cos(x*y)
```

---

## 5. The Gradient

The **gradient** stacks all partial derivatives into a single vector:

$$
\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\nabla f$ | Gradient of $f$ | Vector of all first partial derivatives; points in direction of steepest ascent |
| $\frac{\partial f}{\partial x_i}$ | $i$-th component of gradient | Rate of change of $f$ in the $x_i$ direction |
| $\begin{bmatrix} \vdots \end{bmatrix}$ | Column vector | Gradient is always a column vector of length $n$ |
| $\|\nabla f\|$ | Gradient magnitude | Maximum rate of increase of $f$ at the given point |

### Properties

1. **Direction of steepest ascent:** $\nabla f$ points in the direction where $f$ increases fastest.
2. **Magnitude equals rate of increase:** $\|\nabla f\|$ is the maximum rate of change.
3. **Perpendicular to level sets:** $\nabla f$ is orthogonal to the contour surface at every point.

**ML Connection:** Gradient descent updates parameters in the *opposite* direction of the gradient:

$$
\theta_{t+1} = \theta_t - \alpha \nabla_\theta \mathcal{L}(\theta_t)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\theta_t$ | Parameters at iteration $t$ | Current model weights before the update |
| $\theta_{t+1}$ | Updated parameters | New weights after one gradient descent step |
| $\alpha$ | Learning rate (step size) | Controls how large a step to take; too large = divergence, too small = slow convergence |
| $\nabla_\theta \mathcal{L}(\theta_t)$ | Gradient of loss w.r.t. $\theta$ at $\theta_t$ | Direction of steepest ascent of the loss; moving opposite minimizes loss |
| $-\alpha \nabla_\theta \mathcal{L}(\theta_t)$ | Negative gradient step | The actual update — moves parameters downhill to reduce the loss |

This is the single most important equation in deep learning optimization.

```python
import numpy as np

def f(x, y):
    return x**2 + y**2

def gradient(x, y):
    return np.array([2*x, 2*y])

# Gradient descent
theta = np.array([3.0, 4.0])  # starting point
lr = 0.1

for step in range(20):
    grad = gradient(theta[0], theta[1])
    theta = theta - lr * grad

print(f"Converged to: {theta}")  # ≈ [0, 0]
```

---

## 6. Directional Derivative

The **directional derivative** of $f$ in the direction of a unit vector $\mathbf{u}$ is:

$$
D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u} = \|\nabla f\| \cos \theta
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $D_{\mathbf{u}} f$ | Directional derivative of $f$ in direction $\mathbf{u}$ | Rate of change of $f$ when moving in direction $\mathbf{u}$ (unit vector) |
| $\nabla f$ | Gradient of $f$ | The vector of partial derivatives |
| $\mathbf{u}$ | Unit direction vector | Must have $\|\mathbf{u}\| = 1$; specifies the direction of movement |
| $\nabla f \cdot \mathbf{u}$ | Dot product of gradient with direction | Projects the gradient onto the direction $\mathbf{u}$ |
| $\|\nabla f\| \cos \theta$ | Magnitude times cosine of angle | Maximum when $\mathbf{u}$ aligns with $\nabla f$ ($\theta=0$), zero when perpendicular ($\theta=\pi/2$) |

where $\theta$ is the angle between $\nabla f$ and $\mathbf{u}$.

* Maximum when $\mathbf{u}$ aligns with $\nabla f$ ($\theta = 0$): $D_{\mathbf{u}} f = \|\nabla f\|$.
* Zero when $\mathbf{u}$ is perpendicular to $\nabla f$ ($\theta = \pi/2$): moving along level sets.
* Minimum when $\mathbf{u}$ opposes $\nabla f$ ($\theta = \pi$): steepest descent.

**ML Connection:** This is exactly what gradient descent exploits — by choosing $\mathbf{u} = -\nabla f / \|\nabla f\|$, we move in the direction of steepest descent.

---

## 7. Higher-Order Partial Derivatives

The **Hessian matrix** collects all second-order partial derivatives:

$$
H = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \dots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \dots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \dots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $H$ | Hessian matrix | Square matrix of all second-order partial derivatives |
| $\frac{\partial^2 f}{\partial x_i^2}$ | Pure second partial derivative | Curvature of $f$ in the $x_i$ direction |
| $\frac{\partial^2 f}{\partial x_i \partial x_j}$ | Mixed partial derivative | How the $x_i$-slope changes as $x_j$ varies |
| $H$ symmetric ($H = H^T$) | Clairaut's theorem | Mixed partials are equal for well-behaved $f$; Hessian is symmetric |
| Hessian $\succeq 0$ | Positive semi-definite | Indicates convexity; the loss surface curves upward everywhere |

For well-behaved functions, mixed partials are equal: $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$ (Clairaut's Theorem), making $H$ symmetric.

**ML Connection:** The Hessian describes the **curvature** of the loss landscape. If $H$ is positive definite, the loss surface curves upward in every direction — you're in a bowl and gradient descent converges. If $H$ has mixed signs, you're on a saddle point (Lecture 10).

---

## 8. Concrete ML Example: Linear Regression Loss

For ordinary least squares with $n$ features and $m$ samples:

$$
\mathcal{L}(\mathbf{w}) = \frac{1}{2m} \sum_{i=1}^{m} \left( \mathbf{w}^T \mathbf{x}_i - y_i \right)^2
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathcal{L}(\mathbf{w})$ | Mean Squared Error (MSE) loss | The average squared difference between predictions and targets |
| $\mathbf{w}$ | Weight vector (parameters) | The $n$-dimensional vector of model coefficients |
| $m$ | Number of training samples | Each sample contributes to the total loss |
| $\mathbf{w}^T \mathbf{x}_i$ | Linear prediction for sample $i$ | The dot product of weights with features — the model's prediction |
| $y_i$ | True target for sample $i$ | The ground-truth value we want to predict |
| $\frac{1}{2m}$ | Normalization factor | The $\frac{1}{2}$ simplifies the derivative; $\frac{1}{m}$ averages over samples |

This is a function of $n$ variables $(w_1, w_2, \dots, w_n)$.

* **Gradient:** $\nabla_\mathbf{w} \mathcal{L} = \frac{1}{m} X^T(X\mathbf{w} - \mathbf{y})$
* **Setting gradient to zero:** $X^T X \mathbf{w} = X^T \mathbf{y}$ — the **Normal Equation** (Lecture 21)
* **Hessian:** $H = \frac{1}{m} X^T X$ — always positive semi-definite, confirming convexity

> **Check your intuition:** If $f(x, y) = x^2 + y^2$, what is the directional derivative at $(1, 1)$ in the direction $\mathbf{u} = \frac{1}{\sqrt{2}}(1, -1)$? *(Answer: $\nabla f = (2, 2)$, so $D_{\mathbf{u}} f = (2)(1/\sqrt{2}) + (2)(-1/\sqrt{2}) = 0$. We're moving along a level set — the contour circle — so the function doesn't change.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 32: Limits and Continuity](Lecture%2032%20Limits%20and%20Continuity.md) — provides the limit framework needed to define partial derivatives
- **Next:** [Lecture 34: Jacobian](Lecture%2034%20Jacobian.md) — builds on partial derivatives to describe how vector-valued multivariate functions change
- **Related:** [Lecture 31: Functions](Lecture%2031%20Functions.md) — the single-variable function foundation extended to n variables
- **Related:** [Lecture 34: Jacobian](Lecture%2034%20Jacobian.md) — assembles partial derivatives into a matrix describing local linear behavior
- **Related:** [Lecture 38: Convex Functions](Lecture%2038%20Convex%20Functions.md) — applies multivariate calculus to characterize convexity and optimization
