## Chain Rule (Multivariable)

*Essential Mathematics for ML — Structured Notes*

---

## 1. Chain Rule: Single Variable Recap

### Motivation and Intuition
Every neural network is a composition of functions: each layer is a function of the previous layer's output. To train the network, we need to compute how the loss changes with respect to the weights in *every* layer, not just the last one. The **Chain Rule** is the mathematical tool that lets us decompose derivatives through arbitrarily deep compositions.

### Single Variable Case

If $z = f(y)$ and $y = g(x)$, then:

$$
\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx} = f'(g(x)) \cdot g'(x)
$$

```python
# Single variable chain rule in action
import numpy as np

def f(y):
    return np.sin(y)

def g(x):
    return x**2

x = 3.0
# Chain rule: d/dx [sin(x^2)] = cos(x^2) * 2x
manual = np.cos(g(x)) * 2 * x
automatic = np.cos(x**2) * 2 * x
print(f"Manual: {manual}, Auto: {automatic}")  # Both ≈ -1.97
```

---

## 2. Two-Variable Chain Rule

### Setup
Let $z = f(x, y)$ where $x = x(t)$ and $y = y(t)$ are both functions of a single variable $t$. After substitution, $z$ becomes a function of $t$ alone.

### Formula
Since $z$ is ultimately a function of one variable, we use the **total derivative**:

$$
\frac{dz}{dt} = \frac{\partial f}{\partial x} \cdot \frac{dx}{dt} + \frac{\partial f}{\partial y} \cdot \frac{dy}{dt}
$$

**Intuition:** The total rate of change of $z$ with respect to $t$ is the sum of two pathways: the direct effect through $x$ plus the direct effect through $y$.

### Worked Example

$$
z = x^2 y, \quad x = t^2, \quad y = \sin(t)
$$

**Step 1:** Substitute to verify $z(t) = (t^2)^2 \sin(t) = t^4 \sin(t)$.

**Step 2:** Compute partials:
$$
\frac{\partial z}{\partial x} = 2xy, \quad \frac{\partial z}{\partial y} = x^2
$$

**Step 3:** Compute total derivative:
$$
\frac{dz}{dt} = 2xy \cdot 2t + x^2 \cdot \cos(t) = 4t^3\sin(t) + t^4\cos(t)
$$

```python
import sympy as sp

t = sp.symbols('t')
x = t**2
y = sp.sin(t)
z = x**2 * y

# Direct differentiation
dz_dt_direct = sp.diff(z, t)  # 4*t**3*sin(t) + t**4*cos(t)

# Chain rule
dz_dx = sp.diff(z.subs({x: sp.Symbol('x'), y: sp.Symbol('y')}), sp.Symbol('x'))
dz_dy = sp.diff(z.subs({x: sp.Symbol('x'), y: sp.Symbol('y')}), sp.Symbol('y'))
dx_dt = sp.diff(x, t)
dy_dt = sp.diff(y, t)
dz_dt_chain = (dz_dx.subs(sp.Symbol('x'), x) * dx_dt +
               dz_dy.subs(sp.Symbol('y'), y) * dy_dt)

print(f"Direct: {dz_dt_direct}")
print(f"Chain:  {dz_dt_chain}")  # Both equal
```

---

## 3. Three-Variable Chain Rule

### Generalization
If $z = f(x, y, w)$ where $x = x(t)$, $y = y(t)$, $w = w(t)$:

$$
\frac{dz}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt} + \frac{\partial f}{\partial w}\frac{dw}{dt}
$$

**Pattern:** Sum over all intermediate variables, multiplying the partial derivative of the outer function by the derivative of the inner function.

---

## 4. Multivariable Chain Rule (Multiple Intermediate Variables)

### The General Case
Let $z = f(x_1, x_2, \dots, x_n)$ where each $x_i = x_i(t_1, t_2, \dots, t_m)$ is a function of $m$ variables.

Then for each $t_j$:

$$
\frac{\partial z}{\partial t_j} = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} \cdot \frac{\partial x_i}{\partial t_j}
$$

### Matrix Form
This is exactly the Jacobian chain rule (Lecture 34):

$$
J_{f \circ g} = J_f \cdot J_g
$$

The Jacobian of a composition is the matrix product of the individual Jacobians.

```python
import numpy as np

# Layer composition: z = f(g(x))
# g: R^3 -> R^4 (linear + ReLU)
# f: R^4 -> R^2 (linear)

W1 = np.random.randn(4, 3)
W2 = np.random.randn(2, 4)
x = np.array([1.0, 2.0, 3.0])

# Forward
g_out = np.maximum(0, W1 @ x)  # ReLU
z = W2 @ g_out

# Jacobian of composition = J_f @ J_g
# J_g = diag(relu'(W1@x)) @ W1
# J_f = W2
# J_total = W2 @ diag(relu'(W1@x)) @ W1
```

---

## 5. Chain Rule with Multiple Intermediate Variables

### Setup
Sometimes a variable depends on multiple intermediate paths. Consider:

$$
z = f(u, v), \quad u = g(x, y), \quad v = h(x, y)
$$

Then:

$$
\frac{\partial z}{\partial x} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial x} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial x}
$$

$$
\frac{\partial z}{\partial y} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial y} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial y}
$$

**Intuition:** Both $u$ and $v$ are affected by $x$, and $z$ depends on both $u$ and $v$. The total effect of $x$ on $z$ is the sum of all paths.

### Worked Example

$$
z = u^2 + v^3, \quad u = xy, \quad v = x + y
$$

**Step 1:** $\frac{\partial f}{\partial u} = 2u$, $\frac{\partial f}{\partial v} = 3v^2$

**Step 2:** $\frac{\partial u}{\partial x} = y$, $\frac{\partial v}{\partial x} = 1$

**Step 3:**
$$
\frac{\partial z}{\partial x} = 2u \cdot y + 3v^2 \cdot 1 = 2xy \cdot y + 3(x+y)^2 = 2xy^2 + 3(x+y)^2
$$

---

## 6. The Deep Learning Connection: Backpropagation

### Network Architecture

Consider a 3-layer network:

$$
\mathbf{h}_1 = \sigma(W_1 \mathbf{x} + \mathbf{b}_1)
$$
$$
\mathbf{h}_2 = \sigma(W_2 \mathbf{h}_1 + \mathbf{b}_2)
$$
$$
\hat{y} = W_3 \mathbf{h}_2 + \mathbf{b}_3
$$
$$
\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2
$$

### Forward Pass (Composition)
$\mathcal{L}$ is a function of $\hat{y}$, which is a function of $\mathbf{h}_2$, which is a function of $\mathbf{h}_1$, which is a function of $\mathbf{x}$. The chain rule decomposes the total derivative:

$$
\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial \mathbf{h}_2} \cdot \frac{\partial \mathbf{h}_2}{\partial \mathbf{h}_1} \cdot \frac{\partial \mathbf{h}_1}{\partial W_1}
$$

### Backward Pass (Chain Rule in Reverse)
Each term is computed by multiplying the upstream gradient by the local Jacobian:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot W_3
$$
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_2} \cdot \text{diag}(\sigma'(\mathbf{z}_2)) \cdot W_2
$$
$$
\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} \cdot \text{diag}(\sigma'(\mathbf{z}_1)) \cdot \mathbf{x}^T
$$

This is literally the chain rule applied layer by layer, in reverse order.

```python
import torch

# Autograd does chain rule automatically
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
W1 = torch.randn(4, 3, requires_grad=True)
W2 = torch.randn(2, 4, requires_grad=True)

# Forward pass (composition)
h1 = torch.relu(x @ W1.T)
y_pred = h1 @ W2.T
loss = (y_pred - torch.tensor([1.0, 0.0])).pow(2).sum()

# Backward pass (chain rule applied automatically)
loss.backward()  # This computes dL/dW1, dL/dW2, dL/dx via chain rule
print(f"dL/dW1 shape: {W1.grad.shape}")  # (4, 3)
print(f"dL/dW2 shape: {W2.grad.shape}")  # (2, 4)
```

---

## 7. Deep Learning Failure Modes Related to the Chain Rule

### Vanishing Gradients
In the chain rule, gradients are **multiplied** through each layer. If each layer's Jacobian has small entries ($< 1$), the product shrinks exponentially:

$$
\prod_{i=1}^{L} \|\sigma'(z_i) \cdot W_i\| \to 0 \quad \text{as } L \to \infty
$$

**Fix:** Use ReLU activations ($\sigma'(z) = 1$ for $z > 0$), batch normalization, residual connections.

### Exploding Gradients
If each layer's Jacobian has large entries ($> 1$), the product grows exponentially:

$$
\prod_{i=1}^{L} \|\sigma'(z_i) \cdot W_i\| \to \infty
$$

**Fix:** Gradient clipping, weight decay, orthogonal initialization.

### The ReLU Advantage
For ReLU, $\sigma'(z) = 1$ when $z > 0$. The chain rule product becomes:

$$
\prod_{i=1}^{L} \mathbf{1}_{z_i > 0} \cdot W_i
$$

The gradient neither shrinks nor grows through active neurons — it flows cleanly through the chain. This is why ReLU became the default activation function.

---

## 8. Summary

| Scenario | Formula |
|:---|:---|
| **Single variable** | $\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$ |
| **Two intermediate vars** | $\frac{dz}{dt} = \frac{\partial z}{\partial x}\frac{dx}{dt} + \frac{\partial z}{\partial y}\frac{dy}{dt}$ |
| **General (Jacobian form)** | $J_{f \circ g} = J_f \cdot J_g$ |
| **Backpropagation** | Multiply upstream gradient by local Jacobian at each layer |

> **Check your intuition:** If a network has 50 layers and the Jacobian at each layer has a spectral norm of $0.9$, what happens to the gradient magnitude after backpropagating through all 50 layers? *(Answer: It shrinks by $0.9^{50} \approx 0.005$. The gradient is essentially gone — this is the vanishing gradient problem in its purest form.)*
