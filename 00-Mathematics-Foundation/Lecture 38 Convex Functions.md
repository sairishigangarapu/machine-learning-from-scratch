## Convex Functions

*Essential Mathematics for ML — Structured Notes*

---

## 1. Definition of a Convex Function

### Motivation and Intuition
The single most important property of a convex function is this: **every local minimum is a global minimum**. This means gradient descent cannot get trapped in a suboptimal solution. Linear regression, logistic regression, and SVMs all use convex loss functions, which is why they are guaranteed to converge to the best solution. Neural networks use non-convex loss functions, which is why training them is an art.

### Formal Definition
A function $f: S \subseteq \mathbb{R}^n \to \mathbb{R}$ (where $S$ is a convex set) is **convex** if for all $\mathbf{x}_1, \mathbf{x}_2 \in S$ and $\lambda \in [0, 1]$:

$$
f(\lambda \mathbf{x}_1 + (1 - \lambda)\mathbf{x}_2) \le \lambda f(\mathbf{x}_1) + (1 - \lambda) f(\mathbf{x}_2)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $f$ | Convex function | A function where the line segment between any two points on the graph lies above the graph |
| $\mathbf{x}_1, \mathbf{x}_2 \in S$ | Two points in the domain | The inequality must hold for ALL pairs in the convex set $S$ |
| $\lambda \in [0, 1]$ | Convex combination weight | Interpolates between the two points — $\lambda=0$ gives $\mathbf{x}_2$, $\lambda=1$ gives $\mathbf{x}_1$ |
| $\lambda \mathbf{x}_1 + (1 - \lambda)\mathbf{x}_2$ | Point on line segment | A point between $\mathbf{x}_1$ and $\mathbf{x}_2$ |
| $f(\lambda \mathbf{x}_1 + (1 - \lambda)\mathbf{x}_2)$ | Function value at interpolated point | The actual function value at the interpolated point |
| $\lambda f(\mathbf{x}_1) + (1 - \lambda) f(\mathbf{x}_2)$ | Interpolated function values | The weighted average of the function values at the endpoints |
| $\le$ | Inequality direction | The function value at the interpolated point is ≤ the interpolated function values — "curves upward" |

**Geometric meaning:** The line segment connecting any two points on the graph of $f$ lies **above** or on the graph. The function "curves upward" everywhere.

### Strict Convexity

$$
f(\lambda \mathbf{x}_1 + (1 - \lambda)\mathbf{x}_2) < \lambda f(\mathbf{x}_1) + (1 - \lambda) f(\mathbf{x}_2) \quad \text{for } \lambda \in (0,1), \; \mathbf{x}_1 \neq \mathbf{x}_2
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $<$ | Strict inequality | The function value is strictly less than the chord — no flat linear segments |
| $\lambda \in (0,1)$ | Open interval | Excludes endpoints — strict convexity only required for interior points |
| $\mathbf{x}_1 \neq \mathbf{x}_2$ | Distinct points | Only applies when the two points are different |

Strict convexity guarantees a **unique** global minimum.

---

## 2. First-Order Condition (Differentiable Functions)

If $f$ is differentiable, $f$ is convex if and only if for all $\mathbf{x}, \mathbf{y}$ in its domain:

$$
f(\mathbf{y}) \ge f(\mathbf{x}) + \nabla f(\mathbf{x})^T (\mathbf{y} - \mathbf{x})
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $f(\mathbf{y})$ | Function value at $\mathbf{y}$ | The value we're lower-bounding |
| $f(\mathbf{x})$ | Function value at $\mathbf{x}$ | The point where we construct the tangent |
| $\nabla f(\mathbf{x})$ | Gradient at $\mathbf{x}$ | The vector of partial derivatives — the slope of the tangent hyperplane |
| $\nabla f(\mathbf{x})^T (\mathbf{y} - \mathbf{x})$ | Linear approximation | The directional derivative — how much the tangent predicts the function will increase |
| $f(\mathbf{x}) + \nabla f(\mathbf{x})^T (\mathbf{y} - \mathbf{x})$ | Tangent hyperplane | The first-order Taylor approximation at $\mathbf{x}$ |

**Geometric meaning:** The tangent hyperplane at any point lies **below** the function everywhere. The function never dips below its own linear approximation.

**ML Connection:** This is exactly what gradient descent exploits. The gradient $\nabla f(\mathbf{x})$ gives a local linear approximation, and the convexity condition guarantees that moving in the negative gradient direction always makes progress toward the minimum.

---

## 3. Second-Order Condition

If $f$ is twice differentiable, $f$ is convex if and only if the **Hessian** is positive semi-definite everywhere:

$$
\nabla^2 f(\mathbf{x}) \succeq 0 \quad \forall \mathbf{x}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\nabla^2 f(\mathbf{x})$ | Hessian matrix of $f$ at $\mathbf{x}$ | Matrix of second partial derivatives — measures curvature of $f$ at $\mathbf{x}$ |
| $\succeq 0$ | Positive semi-definite | All eigenvalues are $\ge 0$; the function curves upward or is flat in every direction |
| $\forall \mathbf{x}$ | For all points in the domain | The condition must hold at EVERY point for $f$ to be convex |
| $\succ 0$ (strict) | Positive definite | All eigenvalues $> 0$; the function curves strictly upward everywhere |

For **strict** convexity: $\nabla^2 f(\mathbf{x}) \succ 0$ (positive definite).

```python
import numpy as np

def hessian_check_convexity(f, point, h=1e-5):
    """Numerically approximate the Hessian and check positive semi-definiteness."""
    n = len(point)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            e_i = np.zeros(n); e_i[i] = h
            e_j = np.zeros(n); e_j[j] = h
            H[i,j] = (f(point + e_i + e_j) - f(point + e_i - e_j)
                      - f(point - e_i + e_j) + f(point - e_i - e_j)) / (4 * h**2)
    
    eigenvalues = np.linalg.eigvalsh(H)
    return np.all(eigenvalues >= -1e-10), eigenvalues

# f(x, y) = x^2 + y^2 (convex)
f = lambda v: v[0]**2 + v[1]**2
is_conv, eigs = hessian_check_convexity(f, np.array([1.0, 1.0]))
print(f"Convex: {is_conv}, Eigenvalues: {eigs}")  # True, [2, 2]
```

---

## 4. The Killer Property: Local = Global

### Theorem
If $f$ is convex on a convex set $S$, then **every local minimum is a global minimum**.

### Proof (By Contradiction)

Suppose $\bar{x}$ is a local minimum but NOT a global minimum. Then there exists $\mathbf{x}^* \in S$ with $f(\mathbf{x}^*) < f(\bar{x})$.

By convexity, for $\lambda \in (0, 1)$:

$$
f(\lambda \mathbf{x}^* + (1-\lambda)\bar{x}) \le \lambda f(\mathbf{x}^*) + (1-\lambda) f(\bar{x}) < \lambda f(\bar{x}) + (1-\lambda) f(\bar{x}) = f(\bar{x})
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\lambda \mathbf{x}^* + (1-\lambda)\bar{x}$ | Convex combination of $\mathbf{x}^*$ and $\bar{x}$ | A point on the line segment connecting the two points |
| $f(\lambda \mathbf{x}^* + (1-\lambda)\bar{x})$ | Function at interpolated point | Convexity says this is $\le$ the interpolated function values |
| $\lambda f(\mathbf{x}^*) + (1-\lambda) f(\bar{x})$ | Interpolated function values | The convex combination of the two function values |
| $< f(\bar{x})$ | Strict inequality | Since $f(\mathbf{x}^*) < f(\bar{x})$, interpolated value is below $f(\bar{x})$ |
| Local = Global | Contradiction proof | Points arbitrarily close to $\bar{x}$ have lower $f$, contradicting local minimality |

This means points arbitrarily close to $\bar{x}$ (for small $\lambda$) have **lower** function values, contradicting that $\bar{x}$ is a local minimum. $\blacksquare$

**Deep Learning Connection:** This is why convex optimization problems (linear regression, logistic regression, SVM) are "easy" — you can start gradient descent anywhere and are guaranteed to find the global optimum. Neural network loss functions are **non-convex**, so this guarantee vanishes. Different random initializations can lead to wildly different solutions.

---

## 5. Common Convex Functions

| Function | Formula | Domain | Convexity |
|:---|:---|:---|:---|
| **Linear** | $f(x) = a^T x + b$ | $\mathbb{R}^n$ | Convex (and concave) |
| **Quadratic** | $f(x) = \frac{1}{2}x^T Q x + a^T x + b$, $Q \succeq 0$ | $\mathbb{R}^n$ | Convex |
| **Huber Loss** | Defined piecewise | $\mathbb{R}$ | Convex |
| **ReLU** | $f(x) = \max(0, x)$ | $\mathbb{R}$ | Convex |
| **Log-Sum-Exp** | $f(x) = \log \sum e^{x_i}$ | $\mathbb{R}^n$ | Convex |

### Non-Convex Functions (ML)

| Function | Formula | Why Non-Convex |
|:---|:---|:---|
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | Not convex on $\mathbb{R}$ (S-shaped) |
| **Softmax cross-entropy** | $\mathcal{L} = -\log p_{y}$ | Non-convex in general (but convex in logits for binary case) |
| **Neural network loss** | Composed layers | Non-convex in parameters |

---

## 6. Convexity of Common Loss Functions

### Mean Squared Error (MSE)
$$
\mathcal{L}(\mathbf{w}) = \frac{1}{n}\|X\mathbf{w} - \mathbf{y}\|^2
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathcal{L}(\mathbf{w})$ | Mean Squared Error loss | Average squared residual between predictions and targets |
| $\mathbf{w}$ | Weight vector | Model parameters to be learned |
| $X$ | Design matrix ($n$ samples $\times$ $d$ features) | Each row is a training example |
| $\mathbf{y}$ | Target vector | Ground-truth values for all $n$ samples |
| $\|X\mathbf{w} - \mathbf{y}\|^2$ | Squared residual norm | Sum of squared differences: $\sum_{i=1}^n (\mathbf{w}^T\mathbf{x}_i - y_i)^2$ |
| $\frac{1}{n}$ | Normalization by sample count | Gives average (mean) squared error |

* **Hessian:** $H = \frac{2}{n}X^TX$ — always positive semi-definite.
* **Convex?** Yes. Strictly convex if $X$ has full column rank.
* **Implication:** Linear regression always has a unique global minimum (the Normal Equation).

### Cross-Entropy Loss
$$
\mathcal{L}(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^n [y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathcal{L}(\mathbf{w})$ | Cross-entropy loss | Standard loss for binary classification tasks |
| $n$ | Number of training samples | Loss is averaged over all samples |
| $y_i \in \{0, 1\}$ | True binary label | Ground-truth class for sample $i$ |
| $\hat{y}_i = \sigma(\mathbf{w}^T\mathbf{x}_i)$ | Predicted probability | Model's output after sigmoid — $\hat{y}_i \in (0, 1)$ |
| $y_i \log \hat{y}_i$ | Positive class contribution | Only active when $y_i = 1$ (since $y_i = 0$ zeroes this term) |
| $(1-y_i)\log(1-\hat{y}_i)$ | Negative class contribution | Only active when $y_i = 0$ |

* **Convex?** Yes in the logits (before sigmoid). After sigmoid, the composition can be non-convex in the parameters.
* **Implication:** Logistic regression training converges to the global optimum.

---

## 7. Operations That Preserve Convexity

| Operation | Result |
|:---|:---|
| **Non-negative weighted sum** | $\sum \alpha_i f_i$ is convex if $\alpha_i \ge 0$ and each $f_i$ is convex |
| **Composition with affine** | $g(\mathbf{x}) = f(A\mathbf{x} + \mathbf{b})$ is convex if $f$ is convex |
| **Pointwise maximum** | $h(\mathbf{x}) = \max_i f_i(\mathbf{x})$ is convex if each $f_i$ is convex |
| **Epigraph** | $\text{epi}(f) = \{(\mathbf{x}, t) : f(\mathbf{x}) \le t\}$ is convex if $f$ is convex |

**ML Connection:** The maximum of convex functions (like max-pooling in CNNs) preserves convexity. However, neural network composition through multiple non-linear layers destroys convexity.

---

## 8. Summary

| Concept | Key Takeaway |
|:---|:---|
| **Convex function** | Line segment on graph lies above the graph |
| **First-order condition** | Tangent hyperplane lies below the function |
| **Second-order condition** | Hessian is positive semi-definite |
| **Local = Global** | Every local minimum is THE global minimum |
| **MSE loss** | Convex → Linear regression is "solved" |
| **Neural network loss** | Non-convex → Training is an art |

> **Check your intuition:** Is $f(x) = x^3$ convex on $\mathbb{R}$? *(Answer: No. $f''(x) = 6x$, which is negative for $x < 0$ and positive for $x > 0$. The Hessian is not positive semi-definite everywhere, so the function is not convex. It is convex on $[0, \infty)$ but not on $\mathbb{R}$.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 37: Convex Sets](Lecture%2037%20Convex%20Sets.md) — provides the geometric foundation (convex domains) required for defining convex functions
- **Next:** [Lecture 39: Definiteness of Matrices](Lecture%2039%20Definiteness%20of%20Matrices.md) — classifies the Hessian to test the second-order condition for convexity
- **Related:** [Lecture 37: Convex Sets](Lecture%2037%20Convex%20Sets.md) — convex functions are defined on convex sets; their epigraphs are convex sets
- **Related:** [Lecture 41: Unconstrained Optimization](Lecture%2041%20Unconstrained%20Optimization.md) — convexity guarantees that local minima are global minima in unconstrained problems
- **Related:** [Lecture 42: Constrained Optimization-I](Lecture%2042%20Constrained%20Optimization-I.md) — KKT conditions are sufficient for global optimality when the problem is convex
