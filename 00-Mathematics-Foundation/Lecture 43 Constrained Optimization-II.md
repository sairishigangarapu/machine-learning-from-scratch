## Numerical Optimization Algorithms

*Essential Mathematics for ML — Structured Notes*

---

## 1. Why Numerical Methods?

### Motivation and Intuition
We have the KKT conditions and analytical methods for solving optimization problems. But in practice, most ML models have millions of parameters, non-convex loss landscapes, and no closed-form solution. We cannot solve $\nabla f(\mathbf{x}) = \mathbf{0}$ analytically for a 10-billion-parameter neural network. **Numerical optimization** algorithms iteratively move through the parameter space, improving the solution step by step.

### The General Framework

Every iterative optimization algorithm follows the same template:

$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{d}_k
$$

where:
* $\mathbf{x}_k$: Current point (current model parameters)
* $\mathbf{d}_k$: Search direction (which way to move)
* $\alpha_k$: Step size / learning rate (how far to move)

The entire art of optimization is choosing $\mathbf{d}_k$ and $\alpha_k$ wisely.

```python
# The skeleton of every optimization loop
def optimize(f, grad_f, x0, direction_fn, step_fn, max_iter=1000, tol=1e-8):
    x = x0.copy()
    for k in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        d = direction_fn(g, k)       # choose direction
        alpha = step_fn(x, d, k)     # choose step size
        x = x + alpha * d
    return x
```

---

## 2. Gradient Descent (Steepest Descent)

### Direction: Negative Gradient
The simplest and most fundamental choice:

$$
\mathbf{d}_k = -\nabla f(\mathbf{x}_k)
$$

The negative gradient is the direction of **steepest descent** — it points in the direction where $f$ decreases fastest.

### Convergence Properties

| Property | Rate |
|:---|:---|
| **Convex, Lipschitz gradient** | $O(1/k)$ |
| **Strongly convex** | Linear convergence |
| **Non-convex** | Converges to stationary point (not guaranteed to be global min) |

### Limitations
* **Oscillation:** In narrow valleys, gradient descent bounces back and forth across the valley floor, making slow progress along the valley.
* **Slow convergence near minima:** The gradient approaches zero, so steps become tiny.

---

## 3. Line Search: Choosing Step Size

### Exact Line Search
Choose $\alpha_k$ to minimize $f$ along the direction $\mathbf{d}_k$:

$$
\alpha_k = \arg\min_{\alpha > 0} f(\mathbf{x}_k + \alpha \mathbf{d}_k)
$$

Computationally expensive — requires evaluating $f$ at many candidate points.

### Backtracking Line Search (Armijo Rule)
Start with a large $\alpha$ and shrink until the **sufficient decrease** condition is met:

$$
f(\mathbf{x}_k + \alpha \mathbf{d}_k) \le f(\mathbf{x}_k) + c \alpha \nabla f(\mathbf{x}_k)^T \mathbf{d}_k
$$

where $c \in (0, 1)$ is a small constant (typically $c = 10^{-4}$).

```python
def backtracking_line_search(f, grad_f, x, d, alpha=1.0, c=1e-4, rho=0.9):
    """Backtracking line search with Armijo condition."""
    fx = f(x)
    g = grad_f(x)
    directional_deriv = g @ d
    
    while f(x + alpha * d) > fx + c * alpha * directional_deriv:
        alpha *= rho
    
    return alpha
```

---

## 4. Momentum-Based Methods

### The Problem with Pure Gradient Descent
In ill-conditioned problems (high condition number $\kappa$), gradient descent oscillates. Momentum smooths out these oscillations.

### Heavy Ball Method (Polyak, 1964)

$$
\mathbf{v}_{k+1} = \beta \mathbf{v}_k - \alpha \nabla f(\mathbf{x}_k)
$$
$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{v}_{k+1}
$$

where $\beta \in [0, 1)$ is the momentum coefficient.

**Intuition:** The optimizer builds up "velocity" in consistent directions and slows down in oscillating directions — like a ball rolling down a hill.

### Nesterov Accelerated Gradient

$$
\mathbf{v}_{k+1} = \beta \mathbf{v}_k - \alpha \nabla f(\mathbf{x}_k + \beta \mathbf{v}_k)
$$
$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{v}_{k+1}
$$

**Key difference:** Evaluates the gradient at the *lookahead* position $\mathbf{x}_k + \beta \mathbf{v}_k$, not at $\mathbf{x}_k$. This "look ahead" provides faster convergence: $O(1/k^2)$ for convex functions vs $O(1/k)$ for standard gradient descent.

```python
def gradient_descent_with_momentum(f, grad_f, x0, lr=0.01, beta=0.9, max_iter=1000):
    x = x0.copy()
    v = np.zeros_like(x)
    
    for k in range(max_iter):
        g = grad_f(x)
        v = beta * v - lr * g
        x = x + v
    
    return x
```

---

## 5. Convergence Criteria

How do we know when to stop? Common stopping criteria:

| Criterion | Formula | Meaning |
|:---|:---|:---|
| **Gradient norm** | $\|\nabla f(\mathbf{x}_k)\| < \epsilon$ | Near a stationary point |
| **Function change** | $|f(\mathbf{x}_{k+1}) - f(\mathbf{x}_k)| < \epsilon$ | Loss stopped improving |
| **Parameter change** | $\|\mathbf{x}_{k+1} - \mathbf{x}_k\| < \epsilon$ | Parameters stopped moving |
| **Max iterations** | $k > K$ | Budget exhausted |

```python
def convergence_check(x_old, x_new, f_old, f_new, grad, tol=1e-8):
    """Check multiple convergence criteria."""
    if np.linalg.norm(grad) < tol:
        return "Gradient converged"
    if abs(f_new - f_old) < tol:
        return "Function value converged"
    if np.linalg.norm(x_new - x_old) < tol:
        return "Parameters converged"
    return "Not converged"
```

---

## 6. Classification of Optimization Methods

| Method | Information Used | Per-Iteration Cost | Examples |
|:---|:---|:---|:---|
| **Zero-order** | Function values only | Cheap but slow | Random search, Nelder-Mead |
| **First-order** | Gradient | Moderate | SGD, Adam, RMSprop |
| **Second-order** | Gradient + Hessian | Expensive but fast | Newton, L-BFGS, Natural Gradient |

**Deep Learning:** Almost exclusively first-order methods. Second-order methods require computing and inverting the Hessian ($O(n^2)$ storage, $O(n^3)$ inversion), which is impossible for millions of parameters.

---

## 7. Summary

| Concept | Key Takeaway |
|:---|:---|
| **General framework** | $\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{d}_k$ |
| **Gradient descent** | $\mathbf{d}_k = -\nabla f$, simple but oscillates |
| **Momentum** | Accumulates velocity, smooths oscillations |
| **Line search** | Determines step size $\alpha_k$ |
| **Convergence** | Stop when gradient norm is small enough |

> **Check your intuition:** Why does momentum help in narrow valleys? *(Answer: In a narrow valley, the gradient has a large component across the valley and a small component along it. Without momentum, the optimizer bounces across the valley. With momentum, the across-valley components cancel out (alternating signs) while the along-valley components accumulate, so the optimizer moves faster along the valley floor.)*
