## Unconstrained Optimization

*Essential Mathematics for ML — Structured Notes*

---

## 1. The Optimization Problem

### Motivation and Intuition
Every time you train a machine learning model, you are solving an optimization problem. You have a loss function that measures how bad your model's predictions are, and you want to find the model parameters that minimize this loss. The mathematical framework for understanding *when* a solution exists, *what* conditions it must satisfy, and *how* to find it starts here — with unconstrained optimization.

### Formal Definition
An **unconstrained optimization problem** is:

$$
\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})
$$

where $f: \mathbb{R}^n \to \mathbb{R}$ is the **objective function**.

* If $C = \mathbb{R}^n$ (no restrictions on $\mathbf{x}$), the problem is unconstrained.
* If $C \subset \mathbb{R}^n$, the problem is constrained (Lecture 42).

### Key Terminology

| Term | Definition |
|:---|:---|
| **Feasible point** | Any $\mathbf{x} \in \mathbb{R}^n$ (unconstrained: all points are feasible) |
| **Local minimum** | $\bar{\mathbf{x}}$ where $f(\bar{\mathbf{x}}) \le f(\mathbf{x})$ for all $\mathbf{x}$ in a neighborhood of $\bar{\mathbf{x}}$ |
| **Global minimum** | $\bar{\mathbf{x}}$ where $f(\bar{\mathbf{x}}) \le f(\mathbf{x})$ for ALL $\mathbf{x} \in \mathbb{R}^n$ |
| **Optimal point** | A point that achieves the global minimum |
| **Optimal value** | $f(\bar{\mathbf{x}})$ at the optimal point |

```python
import numpy as np

# f(x) = x^4 - 8x^2 + 5
# Has local minima at x = ±2, local maximum at x = 0
f = lambda x: x**4 - 8*x**2 + 5
```

---

## 2. First-Order Necessary Condition (FONC)

### Theorem
If $f$ is differentiable and $\bar{\mathbf{x}}$ is a local minimum of $f$, then:

$$
\nabla f(\bar{\mathbf{x}}) = \mathbf{0}
$$

**Intuition:** At a local minimum, the gradient must be zero — the function is "flat" in every direction. If the gradient pointed somewhere, you could move that way and decrease the function further, contradicting the local minimum.

A point where $\nabla f = \mathbf{0}$ is called a **critical point** (or stationary point).

**Important:** The FONC is necessary but NOT sufficient. A critical point could be a local minimum, local maximum, or saddle point.

```python
import sympy as sp

x = sp.Symbol('x')
f = x**4 - 8*x**2 + 5

# Find critical points
f_prime = sp.diff(f, x)
critical = sp.solve(f_prime, x)
print(f"Critical points: {critical}")  # [-2, 0, 2]
```

---

## 3. Second-Order Sufficient Condition (SOSC)

### Theorem
If $f$ is twice differentiable and at a critical point $\bar{\mathbf{x}}$ (where $\nabla f(\bar{\mathbf{x}}) = \mathbf{0}$):

$$
\nabla^2 f(\bar{\mathbf{x}}) \succ 0 \quad \text{(Hessian is positive definite)}
$$

then $\bar{\mathbf{x}}$ is a **strict local minimum**.

### Classification of Critical Points

| Hessian at Critical Point | Classification |
|:---|:---|
| $\nabla^2 f \succ 0$ (PD) | Strict local minimum |
| $\nabla^2 f \succeq 0$ (PSD, singular) | Possible minimum (need higher-order tests) |
| $\nabla^2 f \prec 0$ (ND) | Strict local maximum |
| $\nabla^2 f$ indefinite | Saddle point |

**Deep Learning Connection:** This is why positive definite Hessians matter. If the Hessian at your current parameter is PD, you're in a bowl and local descent will find the bottom. If it's indefinite, you're on a saddle point and gradient descent may wander aimlessly.

```python
import numpy as np

def classify_critical_point(f_expr, x_sym, critical_point):
    """Classify a critical point using the Hessian."""
    f_prime = sp.diff(f_expr, x_sym)
    f_double_prime = sp.diff(f_expr, x_sym, 2)
    
    hessian_val = float(f_double_prime.subs(x_sym, critical_point))
    
    if hessian_val > 0:
        return "Local Minimum"
    elif hessian_val < 0:
        return "Local Maximum"
    else:
        return "Inconclusive (use higher-order tests)"

x = sp.Symbol('x')
f = x**4 - 8*x**2 + 5
for cp in [-2, 0, 2]:
    result = classify_critical_point(f, x, cp)
    print(f"x = {cp}: {result}")
```

---

## 4. Global vs Local Minima

For **convex functions** (Lecture 38), the story is beautifully simple:

$$
\text{Local minimum} \implies \text{Global minimum}
$$

For **non-convex functions** (like neural network loss landscapes), local minima may not be global minima.

### The Non-Convex Reality of Deep Learning

In high-dimensional non-convex optimization:

* **Local minima are rare** — most critical points are saddle points.
* **Saddle points dominate** — in $d$ dimensions, a critical point is a saddle point with probability $\sim 1 - 2^{-d}$. For $d = 1000$, essentially ALL critical points are saddle points.
* **Good local minima are often "good enough"** — in overparameterized networks, most local minima have similar loss values.

```python
import numpy as np

# Demonstration: random symmetric matrix is almost certainly indefinite
# (saddle point) in high dimensions
def probability_saddle(d, trials=10000):
    count = 0
    for _ in range(trials):
        H = np.random.randn(d, d)
        H = (H + H.T) / 2  # symmetric
        eigenvalues = np.linalg.eigvalsh(H)
        if np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
            count += 1
    return count / trials

for d in [3, 10, 50, 100]:
    p = probability_saddle(d)
    print(f"d={d}: P(saddle point) ≈ {p:.3f}")
```

---

## 5. Existence of Solutions

### Weierstrass Theorem
If $f$ is **continuous** on a **compact** (closed and bounded) set $S$, then $f$ attains its minimum on $S$.

**Why this matters:** In practice, we often add **regularization** (like weight decay) to the loss function, which makes the effective domain bounded and ensures a minimum exists.

$$
\mathcal{L}_{\text{reg}}(\mathbf{w}) = \mathcal{L}(\mathbf{w}) + \lambda \|\mathbf{w}\|^2
$$

The $\lambda \|\mathbf{w}\|^2$ term grows without bound as $\|\mathbf{w}\| \to \infty$, ensuring the minimum exists.

---

## 6. Worked Example

**Problem:** Find and classify all critical points of $f(x, y) = x^3 + y^3 - 3xy$.

**Step 1: Find critical points**

$$
\frac{\partial f}{\partial x} = 3x^2 - 3y = 0 \implies y = x^2
$$

$$
\frac{\partial f}{\partial y} = 3y^2 - 3x = 0 \implies x = y^2
$$

Substituting: $x = (x^2)^2 = x^4$, so $x^4 - x = 0 \implies x(x^3 - 1) = 0$.

**Critical points:** $(0, 0)$ and $(1, 1)$.

**Step 2: Hessian**

$$
H = \begin{bmatrix} 6x & -3 \\ -3 & 6y \end{bmatrix}
$$

**Step 3: Classify**

At $(0, 0)$: $H = \begin{bmatrix} 0 & -3 \\ -3 & 0 \end{bmatrix}$, eigenvalues $\pm 3$ → **saddle point**.

At $(1, 1)$: $H = \begin{bmatrix} 6 & -3 \\ -3 & 6 \end{bmatrix}$, eigenvalues $3, 9$ → **local minimum**.

```python
import sympy as sp

x, y = sp.symbols('x y')
f = x**3 + y**3 - 3*x*y

# Gradient
grad = [sp.diff(f, var) for var in (x, y)]
crit = sp.solve(grad, (x, y))
print(f"Critical points: {crit}")  # [(0, 0), (1, 1)]

# Hessian
H = sp.hessian(f, (x, y))
for point in crit:
    H_val = H.subs([(x, point[0]), (y, point[1])])
    eigs = H_val.eigenvals()
    print(f"At {point}: eigenvalues = {list(eigs.keys())}")
```

---

## 7. Summary

| Condition | What It Tells You |
|:---|:---|
| $\nabla f = \mathbf{0}$ | Necessary for local min (could also be max or saddle) |
| $\nabla^2 f \succ 0$ | Sufficient for strict local minimum |
| $f$ convex + $\nabla f = \mathbf{0}$ | Sufficient for GLOBAL minimum |
| $\nabla^2 f$ indefinite | Saddle point — NOT a minimum |

> **Check your intuition:** For $f(x) = x^3$, the critical point is at $x = 0$. Is it a minimum, maximum, or saddle? *(Answer: None of the above in the strict sense. $f''(0) = 0$, so the second-order test is inconclusive. Since $f$ is increasing on both sides of $x = 0$ (it's an inflection point), it is neither a min nor a max.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 40: Optimality Conditions](Lecture%2040%20Optimality%20Conditions.md) — establishes the first- and second-order conditions that define when a point is optimal
- **Next:** [Lecture 42: Constrained Optimization-I](Lecture%2042%20Constrained%20Optimization-I.md) — extends unconstrained optimization to problems with inequality constraints via KKT conditions
- **Related:** [Lecture 40: Optimality Conditions](Lecture%2040%20Optimality%20Conditions.md) — provides the theoretical conditions that unconstrained optimization algorithms seek to satisfy
- **Related:** [Lecture 43: Constrained Optimization-II](Lecture%2043%20Constrained%20Optimization-II.md) — covers numerical algorithms for iteratively solving optimization problems
- **Related:** [Lecture 44: Steepest Descent Method](Lecture%2044%20Steepest%20Descent%20Method.md) — the primary algorithm for unconstrained optimization in deep learning
