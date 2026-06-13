## Optimality Conditions

*Essential Mathematics for ML — Structured Notes*

---

## 1. What Does "Optimal" Mean?

### Motivation and Intuition
Before we can optimize (minimize or maximize) a function, we need to know what we're looking for. A **global minimum** is the lowest point in the entire domain. A **local minimum** is the lowest point in a neighborhood. ML models find local minima — understanding when they're good enough is critical.

### Definitions

| Term | Definition |
|:---|:---|
| **Global minimum** | $f(\mathbf{x}^*) \le f(\mathbf{x})$ for all $\mathbf{x}$ in the domain |
| **Local minimum** | $f(\mathbf{x}^*) \le f(\mathbf{x})$ for all $\mathbf{x}$ in a neighborhood of $\mathbf{x}^*$ |
| **Saddle point** | Critical point that is neither a local min nor max |

**ML Connection:** Neural network loss surfaces have millions of local minima. Most are nearly as good as the global minimum. The real enemy is saddle points, not local minima.

---

## 2. First-Order Necessary Condition (FONC)

If $\mathbf{x}^*$ is a local extremum of a differentiable function $f$, then:

$$
\nabla f(\mathbf{x}^*) = \mathbf{0}
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\nabla f(\mathbf{x}^*)$ | Gradient of $f$ evaluated at candidate point $\mathbf{x}^*$ | The gradient vector contains all first-order partial derivatives; at an extremum, it must be zero in every direction |
| $\mathbf{x}^*$ | Candidate optimal point | A point satisfying $\nabla f(\mathbf{0})$ is a **critical point** — it could be a minimum, maximum, or saddle point |

Points satisfying this are called **critical points** (or stationary points).

**Intuition:** At a minimum, the gradient must be zero — the function is "flat" in every direction. No direction of steepest descent exists.

**Warning:** FONC is necessary but not sufficient. Zero gradient could be a min, max, or saddle point.

```python
import numpy as np
from scipy.optimize import minimize

def f(x):
    return x[0]**2 + 2*x[1]**2 - 4*x[0] - 2*x[0]*x[1]

def grad_f(x):
    return np.array([2*x[0] - 4 - 2*x[1], 4*x[1] - 2*x[0]])

# Find critical point
result = minimize(f, x0=[0, 0], jac=grad_f, method='BFGS')
print(f"Critical point: {result.x}")
print(f"Gradient at solution: {grad_f(result.x)}")  # ~[0, 0]
```

---

## 3. Second-Order Sufficient Condition (SOSC)

At a critical point ($\nabla f = \mathbf{0}$), the **Hessian matrix** determines the type:

$$
H = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $H$ | Hessian matrix | Square symmetric matrix of all second-order partial derivatives |
| $\frac{\partial^2 f}{\partial x_i^2}$ | Pure second derivative along $x_i$ | Diagonal entries — curvature in each coordinate direction |
| $\frac{\partial^2 f}{\partial x_i \partial x_j}$ | Mixed partial derivative | Off-diagonal entries — how $x_i$ and $x_j$ interact in curvature |
| $H \succ 0$ | Positive definite Hessian | All eigenvalues $> 0$ — strict local minimum |
| $H \prec 0$ | Negative definite Hessian | All eigenvalues $< 0$ — strict local maximum |
| $H$ indefinite | Mixed eigenvalues | Some positive, some negative — saddle point |

| Hessian at Critical Point | Type |
|:---|:---|
| Positive definite ($H \succ 0$) | **Local minimum** |
| Negative definite ($H \prec 0$) | **Local maximum** |
| Indefinite ($H$ has + and - eigenvalues) | **Saddle point** |
| Positive semidefinite ($H \succeq 0$) | Inconclusive (need higher-order tests) |

**Connection to L39:** Definiteness of the Hessian is exactly the matrix definiteness theory from Lecture 39 applied to optimization.

```python
from numpy.linalg import eigvalsh

def hessian_f(x):
    # For f(x,y) = x^2 + 2y^2 - 4x - 2xy
    return np.array([[2, -2],
                     [-2, 4]])

critical = result.x
H = hessian_f(critical)
eigenvalues = eigvalsh(H)
print(f"Hessian eigenvalues: {eigenvalues}")
print(f"Type: {'Local min' if all(eigenvalues > 0) else 'Saddle' if any(eigenvalues > 0) and any(eigenvalues < 0) else 'Local max'}")
```

---

## 4. Gradient as Direction of Steepest Ascent

The gradient $\nabla f(\mathbf{x})$ points in the direction of **maximum increase** of $f$. Therefore, $-\nabla f(\mathbf{x})$ points in the direction of **steepest descent**.

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k)
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\mathbf{x}_k$ | Current point at iteration $k$ | The parameter vector at step $k$ of the descent process |
| $\mathbf{x}_{k+1}$ | Updated point at iteration $k+1$ | The next parameter vector after moving in the descent direction |
| $\alpha$ | Step size / learning rate | Controls how far to move along the negative gradient; too large causes divergence, too small causes slow convergence |
| $\nabla f(\mathbf{x}_k)$ | Gradient of $f$ at $\mathbf{x}_k$ | Points in the direction of steepest ascent; the negative gradient gives the direction of steepest descent |

This is the foundation of gradient descent (Lecture 44).

**Intuition:** If you're standing on a mountain in fog, the gradient tells you which way is steepest uphill. Following $-\nabla f$ takes you downhill fastest.

---

## 5. Convexity and Global Optimality

### Theorem
If $f$ is **convex** and $\nabla f(\mathbf{x}^*) = \mathbf{0}$, then $\mathbf{x}^*$ is a **global minimum**.

**Why this matters for ML:**
* Linear regression with MSE: convex → global minimum guaranteed
* Logistic regression with cross-entropy: convex → global minimum guaranteed
* Neural networks: non-convex → local minima and saddle points

```python
# Convex function: any local min is global
f_convex = lambda x: x[0]**2 + x[1]**2
grad_convex = lambda x: np.array([2*x[0], 2*x[1]])

result = minimize(f_convex, [5, 5], jac=grad_convex, method='BFGS')
print(f"Global minimum: {result.x}")  # [0, 0] regardless of start
```

---

## 6. Summary

| Condition | What It Tells You | Strength |
|:---|:---|:---|
| $\nabla f = \mathbf{0}$ | Critical point (could be min, max, saddle) | Necessary only |
| $\nabla f = \mathbf{0}$ + $H \succ 0$ | Local minimum | Sufficient |
| $\nabla f = \mathbf{0}$ + $H \prec 0$ | Local maximum | Sufficient |
| $\nabla f = \mathbf{0}$ + $H$ indefinite | Saddle point | Sufficient |
| Convex $f$ + $\nabla f = \mathbf{0}$ | **Global minimum** | Strongest |

> **Check your intuition:** In a neural network with 1 million parameters, how many critical points might exist? Are most of them local minima? *(Answer: Potentially exponentially many critical points. Recent research shows most critical points in overparameterized networks are saddle points, not local minima. This is why SGD works — it escapes saddle points easily.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 39: Definiteness of Matrices](Lecture%2039%20Definiteness%20of%20Matrices.md) — provides the eigenvalue-based classification of Hessians used to test optimality conditions
- **Next:** [Lecture 41: Unconstrained Optimization](Lecture%2041%20Unconstrained%20Optimization.md) — applies optimality conditions to formulate and solve unconstrained optimization problems
- **Related:** [Lecture 39: Definiteness of Matrices](Lecture%2039%20Definiteness%20of%20Matrices.md) — the Hessian definiteness test is the practical tool for checking second-order conditions
- **Related:** [Lecture 41: Unconstrained Optimization](Lecture%2041%20Unconstrained%20Optimization.md) — extends these conditions to global optimality via convexity
- **Related:** [Lecture 44: Steepest Descent Method](Lecture%2044%20Steepest%20Descent%20Method.md) — gradient descent iteratively satisfies the first-order necessary condition
