## Python Implementation of Convex Optimization

*Essential Mathematics for ML — Structured Notes*

---

## 1. Why CVXopt?

### Motivation and Intuition
We have spent many lectures building the theory of convex optimization — KKT conditions, duality, penalty methods. But in practice, you don't solve KKT equations by hand. You use a solver. **CVXopt** is a Python library for convex optimization that handles the heavy lifting: it takes your problem specification (objective, constraints) and returns the optimal solution.

### Installation

```bash
pip install cvxopt
```

### What CVXopt Solves
* Linear programming (LP)
* Quadratic programming (QP)
* Second-order cone programming (SOCP)
* Semidefinite programming (SDP)

**ML Connection:** SVMs are QP problems. Lasso and Ridge can be cast as QPs. Portfolio optimization is an LP. CVXopt handles all of these.

---

## 2. Linear Programming with CVXopt

### Problem Formulation
$$
\min \mathbf{c}^T \mathbf{x} \quad \text{s.t.} \quad G\mathbf{x} \le \mathbf{h}, \quad A\mathbf{x} = \mathbf{b}
$$

### Example: Portfolio Allocation

```python
from cvxopt import matrix, solvers

# Minimize cost: c^T x
c = matrix([1.0, 2.0, 3.0])

# Inequality constraints: Gx <= h
G = matrix([[-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 1.0]])
h = matrix([0.0, 0.0, 0.0, 1.0])

# Equality constraints: Ax = b
A = matrix([[1.0, 1.0, 1.0]])
b = matrix([1.0])

sol = solvers.lp(c, G, h, A, b)
print(f"Optimal x: {list(sol['x'])}")
print(f"Optimal cost: {sol['fun']:.4f}")
```

---

## 3. Quadratic Programming with CVXopt

### Problem Formulation
$$
\min \frac{1}{2}\mathbf{x}^T P \mathbf{x} + \mathbf{q}^T \mathbf{x} \quad \text{s.t.} \quad G\mathbf{x} \le \mathbf{h}, \quad A\mathbf{x} = \mathbf{b}
$$

### Example: Ridge Regression as QP

$$
\min_\mathbf{w} \frac{1}{2}\|X\mathbf{w} - \mathbf{y}\|^2 + \lambda\|\mathbf{w}\|^2
$$

Expanding: $\frac{1}{2}\mathbf{w}^T(X^TX + 2\lambda I)\mathbf{w} - (X^T\mathbf{y})^T\mathbf{w} + \text{const}$

So $P = X^TX + 2\lambda I$ and $\mathbf{q} = -X^T\mathbf{y}$.

```python
import numpy as np
from cvxopt import matrix, solvers

np.random.seed(42)
X = np.random.randn(100, 5)
y = X @ np.array([1, 2, 3, 4, 5]) + 0.1 * np.random.randn(100)

lam = 0.1
P = matrix(X.T @ X + 2 * lam * np.eye(5))
q = matrix(-X.T @ y)

# No inequality constraints (unconstrained QP)
# CVXopt requires G and h even if empty — pass dummy arrays of correct shape
G = matrix(np.zeros((1, 5)))  # shape (m, n) where m=0 real constraints
h = matrix(np.zeros(1))
A = matrix(np.zeros((1, 5)))
b = matrix([0.0])

sol = solvers.qp(P, q, G, h, A, b)
w_cvxopt = np.array(sol['x']).flatten()

# Compare with sklearn
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=lam).fit(X, y)
print(f"CVXopt:  {w_cvxopt}")
print(f"Sklearn: {ridge.coef_}")
```

---

## 4. Solving SVM with CVXopt

The dual of the SVM problem is a QP:

$$$
\min_{\alpha} \frac{1}{2}\boldsymbol{\alpha}^T (y_i y_j \mathbf{x}_i^T \mathbf{x}_j) \boldsymbol{\alpha} - \mathbf{1}^T \boldsymbol{\alpha}
$$

$$
\text{s.t.} \quad 0 \le \alpha_i \le C, \quad \sum \alpha_i y_i = 0
$$

```pythonpython
import numpy as np
from cvxopt import matrix, solvers

def svm_cvxopt(X, y, C=1.0):
    """Solve SVM dual using CVXopt."""
    m, n = X.shape
    K = (y[:, None] * X) @ (y[:, None] * X).T  # kernel matrix
    
    P = matrix(K)
    q = matrix(-np.ones(m))
    
    G = matrix(np.vstack([-np.eye(m), np.eye(m)]))
    h = matrix(np.hstack([np.zeros(m), C * np.ones(m)]))
    
    A = matrix(y.reshape(1, -1), tc='d')
    b = matrix([0.0])
    
    sol = solvers.qp(P, q, G, h, A, b)
    alpha = np.array(sol['x']).flatten()
    
    # Recover weights
    w = (alpha * y) @ X
    # Find bias from support vectors (alpha > 0)
    sv = alpha > 1e-5
    b = np.mean(y[sv] - X[sv] @ w)
    
    return w, b, alpha

# Test
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, centers=2, random_state=42)
y = 2 * y - 1  # convert to {-1, +1}

w, b, alpha = svm_cvxopt(X, y, C=1.0)
print(f"Weights: {w}")
print(f"Bias: {b:.4f}")
print(f"Support vectors: {np.sum(alpha > 1e-5)}")
```

---

## 5. Summary

| CVXopt Function | Problem Type | ML Use Case |
|:---|:---|:---|
| `solvers.lp` | Linear programming | Resource allocation, simple baselines |
| `solvers.qp` | Quadratic programming | SVM, Ridge regression |
| `solvers.sdp` | Semidefinite programming | Kernel learning, graph problems |

> **Check your intuition:** Why is SVM a QP and not an LP? *(Answer: The objective involves $\|\mathbf{w}\|^2 = \mathbf{w}^T\mathbf{w}$, which is quadratic in the variables. LP objectives must be linear — they cannot have squared terms.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 45: Newton's and Penalty Function Method](Lecture%2045%20Newton's%20and%20Penalty%20Function%20Method.md) — Builds on theoretical optimization foundations
- **Next:** [Lecture 47: Sets and Basic Operations](Lecture%2047%20Sets%20and%20Basic%20Operations.md) — Mathematical prerequisites for probability theory
- **Related:** [Lecture 36: Python Implementation of Calculus](Lecture%2036%20Python%20Implementation%20of%20Calculus.md) — Python implementation techniques for mathematical concepts
- **Related:** [Lecture 55: Maximum Margin Classification](Lecture%2055%20Maximum%20Margin%20Classification.md) — SVM optimization problem that uses CVXopt
