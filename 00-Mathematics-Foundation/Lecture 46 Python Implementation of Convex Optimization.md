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

### Note: CVXPY — A Modern Alternative

While CVXopt gives you low-level control, **CVXPY** is a higher-level modeling language that lets you write optimization problems in near-mathematical notation. For most ML work, CVXPY is easier to use:

```python
import cvxpy as cp
import numpy as np

# Ridge regression with CVXPY
np.random.seed(42)
X = np.random.randn(100, 5)
y = X @ np.array([1, 2, 3, 4, 5]) + 0.1 * np.random.randn(100)

w = cp.Variable(5)
loss = cp.sum_squares(X @ w - y)
reg = 0.1 * cp.sum_squares(w)
prob = cp.Problem(cp.Minimize(loss + reg))
prob.solve()

print(f"CVXPY: {w.value}")
```

CVXPY handles the conversion to standard form internally and supports a much wider range of problems. Use CVXopt for learning; use CVXPY for production.

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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{c}$ | Cost coefficient vector | Coefficients of the linear objective function |
| $\mathbf{x}$ | Decision variable vector | Variables to be optimized |
| $G$ | Inequality constraint matrix | Encodes $G\mathbf{x} \le \mathbf{h}$ constraints |
| $\mathbf{h}$ | Inequality constraint bound | Right-hand side of inequality constraints |
| $A$ | Equality constraint matrix | Encodes $A\mathbf{x} = \mathbf{b}$ constraints |
| $\mathbf{b}$ | Equality constraint bound | Right-hand side of equality constraints |
| $\text{s.t.}$ | Subject to | Separates the objective from its constraints |

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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{x}$ | Decision variable vector | Variables to be optimized |
| $P$ | Quadratic cost matrix | Encodes curvature and variable interactions in the objective |
| $\mathbf{q}$ | Linear cost vector | Encodes the linear part of the objective |
| $G$ | Inequality constraint matrix | Left-hand side of $G\mathbf{x} \le \mathbf{h}$ |
| $\mathbf{h}$ | Inequality constraint bound | Right-hand side of inequality constraints |
| $A$ | Equality constraint matrix | Left-hand side of $A\mathbf{x} = \mathbf{b}$ |
| $\mathbf{b}$ | Equality constraint bound | Right-hand side of equality constraints |

Unlike LP (which only has a linear term $\mathbf{c}^T \mathbf{x}$), QP adds a quadratic term $\frac{1}{2}\mathbf{x}^T P \mathbf{x}$, enabling curvature and interactions between variables.

### Example: Ridge Regression as QP

$$
\min_\mathbf{w} \frac{1}{2}\|X\mathbf{w} - \mathbf{y}\|^2 + \lambda\|\mathbf{w}\|^2
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{w}$ | Weight/parameter vector | Model coefficients being optimized |
| $X$ | Feature matrix (design matrix) | Training examples as rows, features as columns |
| $\mathbf{y}$ | Target vector | Observed outcomes to predict |
| $\lambda$ | Regularization strength | Controls trade-off between data fit and coefficient penalty |
| $\|\cdot\|^2$ | Squared Euclidean norm | $\|\mathbf{w}\|^2 = \mathbf{w}^T\mathbf{w} = \sum_i w_i^2$ |

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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\boldsymbol{\alpha}$ | Dual variable vector | Lagrange multipliers; nonzero entries identify support vectors |
| $y_i, y_j$ | Class labels for samples $i$ and $j$ | Encode class sign into the kernel matrix |
| $\mathbf{x}_i, \mathbf{x}_j$ | Feature vectors for samples $i$ and $j$ | Input data points whose dot product forms the kernel |
| $\mathbf{1}$ | Vector of all ones | Makes $\mathbf{1}^T\boldsymbol{\alpha} = \sum_i \alpha_i$ the linear term |

$$
\text{s.t.} \quad 0 \le \alpha_i \le C, \quad \sum \alpha_i y_i = 0
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\alpha_i$ | Dual variable for sample $i$ | Support vector weight for the $i$-th training point |
| $C$ | Regularization parameter | Upper bound on $\alpha_i$; penalizes misclassifications |
| $y_i$ | Class label for sample $i$ | Weighted sum $\sum \alpha_i y_i$ must equal zero |

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
