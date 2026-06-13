## The Jacobian

*Essential Mathematics for ML — Structured Notes*

---

## 1. Definition of the Jacobian

### Motivation and Intuition
When we have a function from $\mathbb{R}^n \to \mathbb{R}$, the gradient (a vector) fully describes how the function changes. But what if the function maps $\mathbb{R}^n \to \mathbb{R}^m$? A neural network's hidden layer, for example, takes in $n$ inputs and produces $m$ outputs. We need a way to capture how *each* output depends on *each* input. The **Jacobian matrix** is exactly this: a matrix of all first-order partial derivatives.

### Formal Definition
Let $f: \mathbb{R}^n \to \mathbb{R}^m$ be a vector-valued function:

$$
f(\mathbf{x}) = \begin{bmatrix} f_1(\mathbf{x}) \\ f_2(\mathbf{x}) \\ \vdots \\ f_m(\mathbf{x}) \end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $f(\mathbf{x})$ | Vector-valued function | Maps $\mathbb{R}^n \to \mathbb{R}^m$ — multiple outputs, each depending on all inputs |
| $\mathbf{x}$ | Input vector in $\mathbb{R}^n$ | $n$-dimensional input features |
| $f_i(\mathbf{x})$ | $i$-th component function | Each output $f_i$ is a scalar function of all $n$ input variables |
| $m$ | Number of outputs | Number of rows in the Jacobian; outputs of the function |
| $n$ | Number of inputs | Number of columns in the Jacobian; input dimensions |

The **Jacobian** $J$ (or $Df$) is the $m \times n$ matrix:

$$
J = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \dots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \dots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \dots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $J \in \mathbb{R}^{m \times n}$ | Jacobian matrix | $m$ rows (outputs) $\times$ $n$ columns (inputs) of partial derivatives |
| $\frac{\partial f_i}{\partial x_j}$ | $(i,j)$-th entry | How output $f_i$ changes when input $x_j$ changes |
| Row $i$ | Gradient of $f_i$ | $\nabla f_i^T$ — how the $i$-th output depends on ALL inputs |
| Column $j$ | Sensitivity w.r.t. $x_j$ | How ALL outputs respond to changes in the $j$-th input |

Each **row** is the gradient of one output function. Each **column** describes how one input affects all outputs.

### Special Cases

* If $m = 1$: The Jacobian is a $1 \times n$ row vector — just $\nabla f^T$.
* If $n = 1$: The Jacobian is an $m \times 1$ column vector — the ordinary derivative.
* If $m = n = 1$: The Jacobian is a scalar — the ordinary derivative.

```python
import numpy as np

# f: R^2 -> R^2
def f(x, y):
    return np.array([x**2 * y, np.sin(x + y)])

# Jacobian at (1, 0)
# df1/dx = 2xy, df1/dy = x^2
# df2/dx = cos(x+y), df2/dy = cos(x+y)
J = np.array([
    [2*1*0, 1**2],       # [0, 1]
    [np.cos(1+0), np.cos(1+0)]  # [cos(1), cos(1)]
])
print(J)
```

---

## 2. Geometric Interpretation

The Jacobian describes the **best linear approximation** of a nonlinear function near a point. Near $\mathbf{x}_0$, the function $f$ behaves like:

$$
f(\mathbf{x}) \approx f(\mathbf{x}_0) + J(\mathbf{x}_0)(\mathbf{x} - \mathbf{x}_0)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $f(\mathbf{x})$ | Function value at point $\mathbf{x}$ | The actual nonlinear function output at $\mathbf{x}$ |
| $f(\mathbf{x}_0)$ | Function value at base point $\mathbf{x}_0$ | The starting point of the linear approximation |
| $J(\mathbf{x}_0)$ | Jacobian evaluated at $\mathbf{x}_0$ | The matrix of partial derivatives at the base point — defines the tangent plane |
| $\mathbf{x} - \mathbf{x}_0$ | Displacement from base point | How far and in what direction we move from $\mathbf{x}_0$ |
| $J(\mathbf{x}_0)(\mathbf{x} - \mathbf{x}_0)$ | Linear correction term | First-order Taylor approximation — predicts change in $f$ due to displacement |
| $\approx$ | Approximation | The linearization is accurate only for small displacements near $\mathbf{x}_0$ |

This is the multivariable generalization of the tangent line approximation $f(x) \approx f(a) + f'(a)(x - a)$.

**Geometric meaning:**
* The Jacobian tells you how a small patch of input space gets **stretched**, **rotated**, and **sheared** into output space.
* If the Jacobian is a square matrix ($m = n$), its **determinant** measures the local volume scaling factor.

```python
import numpy as np

# Polar coordinate transformation: (r, theta) -> (x, y)
def polar_to_cartesian(r, theta):
    return np.array([r * np.cos(theta), r * np.sin(theta)])

# Jacobian of the transformation
def jacobian_polar(r, theta):
    return np.array([
        [np.cos(theta), -r * np.sin(theta)],
        [np.sin(theta),  r * np.cos(theta)]
    ])

# At r=2, theta=pi/4
J = jacobian_polar(2, np.pi/4)
det_J = np.linalg.det(J)
print(f"Jacobian:\n{J}")
print(f"det(J) = {det_J}")  # = r = 2 (area scaling factor)
```

---

## 3. The Jacobian and the Chain Rule

### Single Variable Chain Rule (Recap)

If $z = f(y)$ and $y = g(x)$:

$$
\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $z = f(y)$ | Outer function | $z$ depends on $y$ |
| $y = g(x)$ | Inner function | $y$ depends on $x$ |
| $\frac{dz}{dy}$ | Derivative of outer w.r.t. $y$ | How $z$ changes as $y$ varies |
| $\frac{dy}{dx}$ | Derivative of inner w.r.t. $x$ | How $y$ changes as $x$ varies |
| $\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$ | Chain rule (scalar) | The derivative of the composition is the product of individual derivatives |

### Multivariable Chain Rule via Jacobians

If $\mathbf{z} = f(\mathbf{y})$ where $\mathbf{y} \in \mathbb{R}^m$, and $\mathbf{y} = g(\mathbf{x})$ where $\mathbf{x} \in \mathbb{R}^n$, then:

$$
J_{f \circ g} = J_f \cdot J_g
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $J_{f \circ g}$ | Jacobian of the composition $f(g(\mathbf{x}))$ | The derivative of the full composed function |
| $J_f$ | Jacobian of the outer function $f$ | Evaluated at $g(\mathbf{x})$; has dimensions $\dim(f) \times \dim(g)$ |
| $J_g$ | Jacobian of the inner function $g$ | Evaluated at $\mathbf{x}$; has dimensions $\dim(g) \times \dim(\mathbf{x})$ |
| $\cdot$ | Matrix multiplication | The chain rule for vector functions is matrix multiplication — this IS backpropagation |

The Jacobian of a composition is the **matrix product** of the individual Jacobians. This is the foundation of **backpropagation** — gradients flow backward through the network by multiplying Jacobian matrices at each layer.

```python
import numpy as np

# Layer 1: y = relu(W1 @ x + b1)
# Layer 2: z = W2 @ y + b2

W1 = np.random.randn(4, 3)  # 3 inputs -> 4 hidden
W2 = np.random.randn(2, 4)  # 4 hidden -> 2 outputs

x = np.array([1.0, 2.0, 3.0])
y = np.maximum(0, W1 @ x)  # ReLU activation
z = W2 @ y

# Jacobian of the full network at x:
# J_total = J_layer2 @ J_layer1
# For linear layers, J_layer = W
# So J_total = W2 @ W1 (simplified, ignoring ReLU)
```

---

## 4. The Jacobian Determinant

For a square Jacobian ($m = n$), the **determinant** measures the local volume change:

$$
\det(J) = \frac{\text{volume of output patch}}{\text{volume of input patch}}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\det(J)$ | Jacobian determinant | Measures how the function scales volumes locally (only for square $J$, $m = n$) |
| $\text{volume of output patch}$ | Volume after transformation | The size of an infinitesimal region after applying $f$ |
| $\text{volume of input patch}$ | Volume before transformation | The size of an infinitesimal region in the input space |
| $\det(J) \neq 0$ | Invertible | The function is locally invertible (inverse function theorem) |
| $\det(J) = 0$ | Singular | The function collapses a dimension; information is lost |

| $|\det(J)|$ | Meaning |
|:---|:---|
| $> 1$ | Expansion — the function stretches space |
| $= 1$ | Volume-preserving — no stretching |
| $< 1$ | Compression — the function shrinks space |
| $= 0$ | Singular — the function collapses a dimension |

**ML Connection:** In normalizing flows (invertible generative models), the change-of-variables formula requires the Jacobian determinant:

$$
p(\mathbf{x}) = p(\mathbf{z}) \left| \det\left(J_{f^{-1}}(\mathbf{x})\right) \right|
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $p(\mathbf{x})$ | Probability density in data space | The complex distribution we want to model (e.g., natural images) |
| $p(\mathbf{z})$ | Probability density in latent space | A simple base distribution (e.g., standard normal $\mathcal{N}(0, I)$) |
| $f^{-1}$ | Inverse transformation (data $\to$ latent) | Maps data points to latent codes |
| $\det(J_{f^{-1}})$ | Jacobian determinant of $f^{-1}$ | Volume change factor — how much the transformation stretches/compresses space |
| $|\det(\cdot)|$ | Absolute determinant | Probabilities must be non-negative; we take absolute value |

The determinant tells us how the probability density changes when we transform between the latent space and the data space. Efficient flow architectures (RealNVP, Glow) are designed so the Jacobian determinant is fast to compute (triangular matrix).

---

## 5. Worked Example: 2D Transformation

**Function:** $f: \mathbb{R}^2 \to \mathbb{R}^2$ defined by:

$$
f(u, v) = \begin{bmatrix} u^2 - v^2 \\ 2uv \end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $f$ | 2D transformation $\mathbb{R}^2 \to \mathbb{R}^2$ | Maps a pair $(u, v)$ to another pair — like a neural network layer |
| $u^2 - v^2$ | First output $f_1(u,v)$ | Real part of the complex square $(u+iv)^2$ |
| $2uv$ | Second output $f_2(u,v)$ | Imaginary part of the complex square $(u+iv)^2$ |
| $z \mapsto z^2$ | Complex squaring | This transformation is the real representation of $f(z) = z^2$ |

(This is related to the complex mapping $z \mapsto z^2$.)

**Step 1: Compute the Jacobian**

$$
J = \begin{bmatrix}
\frac{\partial f_1}{\partial u} & \frac{\partial f_1}{\partial v} \\
\frac{\partial f_2}{\partial u} & \frac{\partial f_2}{\partial v}
\end{bmatrix}
= \begin{bmatrix}
2u & -2v \\
2v & 2u
\end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\frac{\partial f_1}{\partial u} = 2u$ | Partial derivative of $f_1$ w.r.t. $u$ | $f_1 = u^2 - v^2$, so $\partial f_1/\partial u = 2u$ |
| $\frac{\partial f_1}{\partial v} = -2v$ | Partial derivative of $f_1$ w.r.t. $v$ | $f_1 = u^2 - v^2$, so $\partial f_1/\partial v = -2v$ |
| $\frac{\partial f_2}{\partial u} = 2v$ | Partial derivative of $f_2$ w.r.t. $u$ | $f_2 = 2uv$, so $\partial f_2/\partial u = 2v$ |
| $\frac{\partial f_2}{\partial v} = 2u$ | Partial derivative of $f_2$ w.r.t. $v$ | $f_2 = 2uv$, so $\partial f_2/\partial v = 2u$ |

**Step 2: Evaluate at $(u, v) = (1, 1)$**

$$
J(1,1) = \begin{bmatrix} 2 & -2 \\ 2 & 2 \end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $J(1,1)$ | Jacobian evaluated at $(u,v) = (1,1)$ | The local linear approximation of $f$ at the point $(1,1)$ |
| $2$ (top-left) | $\partial f_1/\partial u$ at $(1,1)$ | $2u$ evaluated at $u=1$ gives $2$ |
| $-2$ (top-right) | $\partial f_1/\partial v$ at $(1,1)$ | $-2v$ evaluated at $v=1$ gives $-2$ |
| $2$ (bottom-left) | $\partial f_2/\partial u$ at $(1,1)$ | $2v$ evaluated at $v=1$ gives $2$ |
| $2$ (bottom-right) | $\partial f_2/\partial v$ at $(1,1)$ | $2u$ evaluated at $u=1$ gives $2$ |

**Step 3: Determinant**

$$
\det(J) = (2)(2) - (-2)(2) = 4 + 4 = 8
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $(2)(2)$ | $J_{11} \cdot J_{22}$ | Product of diagonal entries |
| $(-2)(2)$ | $J_{12} \cdot J_{21}$ | Product of off-diagonal entries |
| $\det(J) = 8$ | Determinant for $2\times 2$: $ad - bc$ | $= (2)(2) - (-2)(2) = 4 + 4 = 8$ |
| $\det(J) > 1$ | Area expansion | Input area is stretched by a factor of $8$; $|\det(J)| > 1$ means expansion |

A small patch of area $\epsilon$ near $(1,1)$ gets stretched to area $8\epsilon$ by the transformation.

```python
import numpy as np

def f(u, v):
    return np.array([u**2 - v**2, 2*u*v])

def jacobian(u, v):
    return np.array([
        [2*u, -2*v],
        [2*v,  2*u]
    ])

J = jacobian(1, 1)
print(f"Jacobian at (1,1):\n{J}")
print(f"det(J) = {np.linalg.det(J)}")  # 8.0
```

---

## 6. The Jacobian in Backpropagation

In a neural network with layer $\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})$:

The Jacobian of the $i$-th neuron with respect to its inputs is:

$$
\frac{\partial h_i}{\partial x_j} = \sigma'(z_i) \cdot W_{ij}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $h_i$ | $i$-th neuron output | Activation of the $i$-th neuron in the layer |
| $x_j$ | $j$-th input to the layer | The $j$-th component of the input vector $\mathbf{x}$ |
| $\sigma'(z_i)$ | Derivative of activation function | For sigmoid $\sigma(z)(1-\sigma(z))$, for ReLU $1$ if $z_i > 0$ else $0$ |
| $z_i$ | Pre-activation of $i$-th neuron | Weighted sum $z_i = \mathbf{w}_i^T \mathbf{x} + b_i$ |
| $W_{ij}$ | Weight connecting $x_j$ to neuron $i$ | The $(i,j)$-th entry of the weight matrix |

where $z_i = \mathbf{w}_i^T \mathbf{x} + b_i$ is the pre-activation.

The full Jacobian of the layer is:

$$
J_{\text{layer}} = \text{diag}(\sigma'(\mathbf{z})) \cdot W
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $J_{\text{layer}}$ | Jacobian of the layer | How each output of the layer changes with each input |
| $\text{diag}(\sigma'(\mathbf{z}))$ | Diagonal matrix of activation derivatives | Element-wise: $\sigma'(z_i)$ on diagonal, zeros elsewhere |
| $W$ | Weight matrix | The linear transformation part of the layer: $W\mathbf{x} + \mathbf{b}$ |
| $\cdot$ | Matrix multiplication | Composition of activation derivative (element-wise nonlinearity) and weights (linear transform) |

During backpropagation, the gradient of the loss with respect to the inputs of this layer is computed by multiplying the incoming gradient by this Jacobian:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = J_{\text{layer}}^T \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{h}}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$ | Gradient of loss w.r.t. layer input | The error signal propagated backward to the previous layer |
| $\frac{\partial \mathcal{L}}{\partial \mathbf{h}}$ | Gradient of loss w.r.t. layer output | The incoming error signal from the next layer (upstream gradient) |
| $J_{\text{layer}}^T$ | Transpose of the layer Jacobian | Backpropagation multiplies by $J^T$ (not $J$) to reverse the flow |
| $J_{\text{layer}}^T \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{h}}$ | Gradient propagation | This is the chain rule in matrix form: $\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \left(\frac{\partial \mathbf{h}}{\partial \mathbf{x}}\right)^T \frac{\partial \mathcal{L}}{\partial \mathbf{h}}$ |

**Deep Learning Failure Mode:** If $\sigma'(z_i) \approx 0$ for most neurons (e.g., saturated sigmoid), the Jacobian becomes nearly zero, and gradients vanish as they propagate backward. This is the **Vanishing Gradient Problem** in its purest mathematical form.

---

## 7. Summary

| Concept | Description | ML Role |
|:---|:---|:---|
| **Jacobian** $J$ | Matrix of all first partial derivatives | Describes local linear behavior |
| **Row** $i$ | Gradient of output $f_i$ | How one output depends on all inputs |
| **Column** $j$ | How input $x_j$ affects all outputs | Sensitivity analysis |
| **$\det(J)$** | Local volume scaling factor | Normalizing flows, change of variables |
| **Chain rule** | $J_{f \circ g} = J_f \cdot J_g$ | Backpropagation |

> **Check your intuition:** If the Jacobian of a transformation has $\det(J) = 0$ at some point, what does that mean geometrically? *(Answer: The transformation collapses space at that point — a 2D region gets squashed onto a 1D line or a point. The transformation is not invertible there, and information is lost. In a normalizing flow, this would be catastrophic — the probability density would be undefined.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 33: Functions of N Variables](Lecture%2033%20Functions%20of%20N%20Variables.md) — introduces partial derivatives and gradients that the Jacobian assembles into a matrix
- **Next:** [Lecture 35: Chain Rule](Lecture%2035%20Chain%20Rule.md) — uses Jacobian multiplication to compose derivatives through network layers
- **Related:** [Lecture 33: Functions of N Variables](Lecture%2033%20Functions%20of%20N%20Variables.md) — the multivariate function framework that the Jacobian extends
- **Related:** [Lecture 35: Chain Rule](Lecture%2035%20Chain%20Rule.md) — the Jacobian chain rule is the mathematical foundation of backpropagation
- **Related:** [Lecture 39: Definiteness of Matrices](Lecture%2039%20Definiteness%20of%20Matrices.md) — classifies Hessian matrices (a special Jacobian) to determine curvature of loss surfaces
