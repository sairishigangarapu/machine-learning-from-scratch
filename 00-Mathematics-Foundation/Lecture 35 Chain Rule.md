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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $z = f(y)$ | Outer function | The function applied to the intermediate result |
| $y = g(x)$ | Inner function | The function applied to the input variable $x$ |
| $\frac{dz}{dy}$ | Derivative of outer w.r.t. inner | How $z$ changes as $y$ changes |
| $\frac{dy}{dx}$ | Derivative of inner w.r.t. input | How $y$ changes as $x$ changes |
| $\frac{dz}{dx}$ | Total derivative | The rate of change of the final output with respect to the original input — the product of the two derivatives |

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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $z = f(x, y)$ | Function of two variables | The outer function depending on $x$ and $y$ |
| $\frac{\partial f}{\partial x}$ | Partial derivative w.r.t. $x$ | How $f$ changes when $x$ changes, holding $y$ fixed |
| $\frac{\partial f}{\partial y}$ | Partial derivative w.r.t. $y$ | How $f$ changes when $y$ changes, holding $x$ fixed |
| $\frac{dx}{dt}$ | Derivative of $x$ w.r.t. $t$ | How the first intermediate variable changes with $t$ |
| $\frac{dy}{dt}$ | Derivative of $y$ w.r.t. $t$ | How the second intermediate variable changes with $t$ |
| $\frac{dz}{dt}$ | Total derivative | Sum of all paths through which $t$ influences $z$ — this IS the chain rule |

### Worked Example

$$
z = x^2 y, \quad x = t^2, \quad y = \sin(t)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $z = x^2 y$ | Outer function $f(x, y)$ | The function of two variables whose total derivative we want |
| $x = t^2$ | First intermediate variable | $x$ depends on $t$ via a quadratic |
| $y = \sin(t)$ | Second intermediate variable | $y$ depends on $t$ via a trigonometric function |
| Substitution $z(t) = t^4 \sin(t)$ | Direct composition | Verifying the chain rule by substituting both intermediates |

**Step 1:** Substitute to verify $z(t) = (t^2)^2 \sin(t) = t^4 \sin(t)$.

**Step 2:** Compute partials:
$$
\frac{\partial z}{\partial x} = 2xy, \quad \frac{\partial z}{\partial y} = x^2
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\frac{\partial z}{\partial x} = 2xy$ | Partial derivative w.r.t. $x$ | Differentiate $x^2 y$ treating $y$ as constant: $2x \cdot y$ |
| $\frac{\partial z}{\partial y} = x^2$ | Partial derivative w.r.t. $y$ | Differentiate $x^2 y$ treating $x$ as constant: $x^2 \cdot 1$ |

**Step 3:** Compute total derivative:
$$
\frac{dz}{dt} = 2xy \cdot 2t + x^2 \cdot \cos(t) = 4t^3\sin(t) + t^4\cos(t)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $2xy \cdot 2t$ | First path: $\frac{\partial z}{\partial x} \cdot \frac{dx}{dt}$ | $2xy$ (from partial) $\times$ $2t$ (since $x = t^2$) |
| $x^2 \cdot \cos(t)$ | Second path: $\frac{\partial z}{\partial y} \cdot \frac{dy}{dt}$ | $x^2$ (from partial) $\times$ $\cos(t)$ (since $y = \sin(t)$) |
| $4t^3\sin(t)$ | First path after substitution | Substitute $x = t^2$, $y = \sin(t)$: $2(t^2)(\sin(t))(2t) = 4t^3\sin(t)$ |
| $t^4\cos(t)$ | Second path after substitution | Substitute $x = t^2$: $(t^2)^2 \cos(t) = t^4\cos(t)$ |
| $\frac{dz}{dt}$ | Total derivative | Sum of all paths: $4t^3\sin(t) + t^4\cos(t)$ — matches direct differentiation of $t^4\sin(t)$ |

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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $z = f(x, y, w)$ | Function of three intermediates | The outer function depends on three variables instead of two |
| $\frac{\partial f}{\partial x}\frac{dx}{dt}$ | Path through $x$ | How $z$ changes via the $x$ intermediate |
| $\frac{\partial f}{\partial y}\frac{dy}{dt}$ | Path through $y$ | How $z$ changes via the $y$ intermediate |
| $\frac{\partial f}{\partial w}\frac{dw}{dt}$ | Path through $w$ | How $z$ changes via the $w$ intermediate |
| $+$ | Sum of all paths | The total derivative is the sum over ALL intermediate variables |

**Pattern:** Sum over all intermediate variables, multiplying the partial derivative of the outer function by the derivative of the inner function.

---

## 4. Multivariable Chain Rule (Multiple Intermediate Variables)

### The General Case
Let $z = f(x_1, x_2, \dots, x_n)$ where each $x_i = x_i(t_1, t_2, \dots, t_m)$ is a function of $m$ variables.

Then for each $t_j$:

$$
\frac{\partial z}{\partial t_j} = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} \cdot \frac{\partial x_i}{\partial t_j}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $z = f(x_1, \dots, x_n)$ | Function of $n$ intermediate variables | The outer function (e.g., loss) depending on all intermediate values (e.g., layer outputs) |
| $\frac{\partial f}{\partial x_i}$ | Partial derivative of outer function | How the output changes when the $i$-th intermediate variable changes |
| $x_i = x_i(t_1, \dots, t_m)$ | Intermediate variable as function of $m$ inputs | Each $x_i$ depends on all inputs (e.g., weights, data) |
| $\frac{\partial x_i}{\partial t_j}$ | Partial derivative of intermediate w.r.t. input | How the $i$-th intermediate changes when the $j$-th input changes |
| $\sum_{i=1}^{n}$ | Sum over all intermediate variables | Accounts for ALL paths from input $t_j$ to output $z$ — the total derivative |

---

## 5. Chain Rule with Multiple Intermediate Variables

### Setup
Sometimes a variable depends on multiple intermediate paths. Consider:

$$
z = f(u, v), \quad u = g(x, y), \quad v = h(x, y)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $z = f(u, v)$ | Output depends on two intermediates | The outer function takes $u$ and $v$ as arguments |
| $u = g(x, y)$ | First intermediate function | $u$ depends on both $x$ and $y$ |
| $v = h(x, y)$ | Second intermediate function | $v$ also depends on both $x$ and $y$ |
| Two intermediate paths | Multiple dependency paths | Both $u$ and $v$ connect $x$ to $z$ |

Then:

$$
\frac{\partial z}{\partial x} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial x} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial x}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $z = f(u, v)$ | Function of two intermediates | The output depends on $u$ and $v$ |
| $u = g(x, y), v = h(x, y)$ | Intermediates depend on $x, y$ | Each intermediate depends on both inputs |
| $\frac{\partial f}{\partial u}$ | Partial of outer w.r.t. first intermediate | How output changes with $u$ |
| $\frac{\partial u}{\partial x}$ | Partial of first intermediate w.r.t. $x$ | How $u$ changes with $x$ |
| $\frac{\partial f}{\partial v}\frac{\partial v}{\partial x}$ | Second path contribution | The alternative path from $x$ to $z$ through $v$ |
| $+$ | Sum of paths | Both paths contribute to the total derivative |

---

## 6. The Deep Learning Connection: Backpropagation

### Network Architecture

Consider a 3-layer network:

$$
\begin{aligned}
\mathbf{h}_1 &= \sigma(W_1 \mathbf{x} + \mathbf{b}_1) \\
\mathbf{h}_2 &= \sigma(W_2 \mathbf{h}_1 + \mathbf{b}_2) \\
\hat{y} &= W_3 \mathbf{h}_2 + \mathbf{b}_3 \\
\mathcal{L} &= \frac{1}{2}(\hat{y} - y)^2
\end{aligned}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{x}$ | Network input | Raw feature vector (e.g., image pixels) |
| $W_i, \mathbf{b}_i$ | Weight matrix and bias of layer $i$ | Learnable parameters that transform the input at each layer |
| $\sigma$ | Activation function | Element-wise non-linearity (e.g., ReLU, sigmoid) |
| $\mathbf{h}_i$ | Hidden activations of layer $i$ | Intermediate representations learned by the network |
| $\hat{y}$ | Network prediction | Final output of the forward pass |
| $\mathcal{L}$ | Loss function | Scalar measure of prediction error, e.g., MSE $= \frac{1}{2}(\hat{y} - y)^2$ |

### Forward Pass (Composition)
$\mathcal{L}$ is a function of $\hat{y}$, which is a function of $\mathbf{h}_2$, which is a function of $\mathbf{h}_1$, which is a function of $\mathbf{x}$. The chain rule decomposes the total derivative:

$$
\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial \mathbf{h}_2} \cdot \frac{\partial \mathbf{h}_2}{\partial \mathbf{h}_1} \cdot \frac{\partial \mathbf{h}_1}{\partial W_1}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\frac{\partial \mathcal{L}}{\partial \hat{y}}$ | Gradient of loss w.r.t. output | The "error signal" — how wrong the prediction is |
| $\frac{\partial \hat{y}}{\partial \mathbf{h}_2}$ | Jacobian of output w.r.t. hidden layer | How output changes when hidden activations change — typically $W_3$ |
| $\frac{\partial \mathbf{h}_2}{\partial \mathbf{h}_1}$ | Jacobian of layer 2 w.r.t. layer 1 | How second hidden layer changes with first — includes activation derivative $\sigma'$ |
| $\frac{\partial \mathbf{h}_1}{\partial W_1}$ | Jacobian of layer 1 w.r.t. weights | How first hidden layer changes with weights — typically $\mathbf{x}^T$ |
| Product of all | Chain rule through all layers | The gradient of loss w.r.t. $W_1$ — what we need for weight updates |

### Backward Pass (Chain Rule in Reverse)
Each term is computed by multiplying the upstream gradient by the local Jacobian:

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_2} &= \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot W_3 \\
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} &= \frac{\partial \mathcal{L}}{\partial \mathbf{h}_2} \cdot \text{diag}(\sigma'(\mathbf{z}_2)) \cdot W_2 \\
\frac{\partial \mathcal{L}}{\partial W_1} &= \frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} \cdot \text{diag}(\sigma'(\mathbf{z}_1)) \cdot \mathbf{x}^T
\end{aligned}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\frac{\partial \mathcal{L}}{\partial \mathbf{h}_2}$ | Gradient at layer 2 | The error signal propagated back from the output |
| $W_3$ | Weight matrix of layer 3 | Transposes to propagate gradients backward |
| $\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1}$ | Gradient at layer 1 | Error signal further back — multiplied by activation derivative and $W_2$ |
| $\text{diag}(\sigma'(\mathbf{z}_2))$ | Diagonal matrix of activation derivatives | The ReLU/sigmoid derivative — zeros out gradients for inactive neurons |
| $W_2$ | Weight matrix of layer 2 | Propagates gradients to previous layer |
| $\frac{\partial \mathcal{L}}{\partial W_1}$ | Gradient w.r.t. weights | Final result — what we use to update $W_1$ via gradient descent |
| $\mathbf{x}^T$ | Input transpose | The outer product with the error signal gives the weight gradient |

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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\prod_{i=1}^{L}$ | Product over all $L$ layers | Gradients multiply through every layer in the chain |
| $\|\sigma'(z_i) \cdot W_i\|$ | Spectral norm of layer $i$ Jacobian | Measures how much the gradient shrinks ($<1$) or grows ($>1$) at layer $i$ |
| $\to 0$ | Vanishes as $L \to \infty$ | Product of many entries $<1$ goes to zero exponentially — the vanishing gradient |
| $L$ | Number of layers | Deeper networks suffer more severe vanishing gradients |

**Fix:** Use ReLU activations ($\sigma'(z) = 1$ for $z > 0$), batch normalization, residual connections.

### Exploding Gradients
If each layer's Jacobian has large entries ($> 1$), the product grows exponentially:

$$
\prod_{i=1}^{L} \|\sigma'(z_i) \cdot W_i\| \to \infty
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\prod_{i=1}^{L}$ | Product over all layers | Same chain rule product structure |
| $\|\sigma'(z_i) \cdot W_i\|$ | Per-layer Jacobian norm | If this value $> 1$ at each layer, the product grows exponentially |
| $\to \infty$ | Explodes as $L \to \infty$ | Gradients become astronomically large — causes NaN, divergence |
| Exploding gradient | Unstable training | Weights receive massive updates, overshooting minima |

**Fix:** Gradient clipping, weight decay, orthogonal initialization.

### The ReLU Advantage
For ReLU, $\sigma'(z) = 1$ when $z > 0$. The chain rule product becomes:

$$
\prod_{i=1}^{L} \mathbf{1}_{z_i > 0} \cdot W_i
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{1}_{z_i > 0}$ | Indicator for active neurons | $1$ if pre-activation $z_i > 0$, $0$ otherwise — ReLU derivative |
| $W_i$ | Weight matrix of layer $i$ | The linear part of the layer Jacobian |
| $\mathbf{1}_{z_i > 0} \cdot W_i$ | ReLU layer Jacobian | Activation derivative $=1$ for active neurons (no shrinkage), $=0$ for inactive (no gradient) |
| Clean gradient flow | No exponential shrinkage | Active ReLU neurons pass gradients with magnitude $1$ — vanishing/exploding is mitigated |

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

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 34: Jacobian](Lecture%2034%20Jacobian.md) — provides the matrix formulation of derivatives that the chain rule composes through layers
- **Next:** [Lecture 36: Python Implementation of Calculus](Lecture%2036%20Python%20Implementation%20of%20Calculus.md) — shows how to verify and implement chain rule computations using SymPy and autodiff
- **Related:** [Lecture 34: Jacobian](Lecture%2034%20Jacobian.md) — the Jacobian chain rule $J_{f \circ g} = J_f \cdot J_g$ is the multivariate generalization
- **Related:** [Lecture 44: Steepest Descent Method](Lecture%2044%20Steepest%20Descent%20Method.md) — gradient descent relies on chain rule gradients computed via backpropagation
