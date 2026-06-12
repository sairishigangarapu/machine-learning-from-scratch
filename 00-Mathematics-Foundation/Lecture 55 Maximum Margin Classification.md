## Maximum Margin Classification

*Essential Mathematics for ML — Structured Notes*

---

## 1. Formal Problem Setup

### Motivation and Intuition
We want to find the hyperplane that separates two classes with maximum margin. This is a constrained optimization problem — minimize the norm of the weight vector subject to all points being correctly classified with a minimum functional margin of 1.

### The Primal Problem
Given training data $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ where $y_i \in \{-1, +1\}$:

$$
\min_{\mathbf{w}, b} \quad \frac{1}{2}||\mathbf{w}||^2
$$
$$
\text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1 \quad \forall i = 1, \dots, n
$$

---

## 2. Lagrangian Formulation

### Lagrangian
$$
\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}||\mathbf{w}||^2 - \sum_{i=1}^n \alpha_i \left[ y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1 \right]
$$

where $\alpha_i \ge 0$ are Lagrange multipliers.

### KKT Conditions
The optimal solution must satisfy:

| Condition | Formula | Meaning |
|:---|:---|:---|
| **Stationarity** | $\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \mathbf{w} - \sum \alpha_i y_i \mathbf{x}_i = 0$ | $\mathbf{w} = \sum \alpha_i y_i \mathbf{x}_i$ |
| **Stationarity** | $\frac{\partial \mathcal{L}}{\partial b} = -\sum \alpha_i y_i = 0$ | $\sum \alpha_i y_i = 0$ |
| **Primal feasibility** | $y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1$ | All constraints satisfied |
| **Dual feasibility** | $\alpha_i \ge 0$ | Multipliers non-negative |
| **Complementary slackness** | $\alpha_i[y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1] = 0$ | Either $\alpha_i = 0$ or point is on margin |

**Key Insight from Complementary Slackness:** Only support vectors (points on the margin) have $\alpha_i > 0$. All other points have $\alpha_i = 0$ and don't affect the solution.

```python
import numpy as np

def compute_support_vectors(X, y, alpha, tol=1e-4):
    """Find support vectors where alpha > 0."""
    sv_mask = alpha > tol
    return X[sv_mask], y[sv_mask], alpha[sv_mask]
```

---

## 3. The Dual Problem

### Derivation
Substituting the stationarity conditions into the Lagrangian:

$$
\max_{\boldsymbol{\alpha}} \quad \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j
$$
$$
\text{s.t.} \quad \alpha_i \ge 0 \quad \forall i, \quad \sum_{i=1}^n \alpha_i y_i = 0
$$

### Why the Dual?
* Only depends on dot products $\mathbf{x}_i^T\mathbf{x}_j$ — enables the kernel trick
* Number of variables equals number of data points (not dimensionality)
* Sparsity: most $\alpha_i = 0$

### Solution Recovery
Once $\boldsymbol{\alpha}^*$ is found:
$$
\mathbf{w}^* = \sum_{i=1}^n \alpha_i^* y_i \mathbf{x}_i
$$
$$
b^* = y_s - \mathbf{w}^{*T}\mathbf{x}_s \quad \text{(for any support vector $\mathbf{x}_s$)}
$$

---

## 4. Quadratic Programming

The dual is a **Quadratic Program (QP)**:

$$
\min_{\boldsymbol{\alpha}} \quad \frac{1}{2}\boldsymbol{\alpha}^T Q \boldsymbol{\alpha} - \mathbf{1}^T\boldsymbol{\alpha}
$$
$$
\text{s.t.} \quad \mathbf{y}^T\boldsymbol{\alpha} = 0, \quad \boldsymbol{\alpha} \ge 0
$$

where $Q_{ij} = y_i y_j \mathbf{x}_i^T \mathbf{x}_j$.

```python
import numpy as np
from cvxopt import matrix, solvers

def solve_svm_dual(X, y):
    n = len(y)
    Q = np.outer(y, y) * (X @ X.T)
    
    P = matrix(Q, tc='d')
    q = matrix(-np.ones(n), tc='d')
    G = matrix(-np.eye(n), tc='d')
    h = matrix(np.zeros(n), tc='d')
    A = matrix(y.reshape(1, -1), tc='d')
    b = matrix(0.0, tc='d')
    
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    return np.array(sol['x']).flatten()

# Example usage
X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
y = np.array([1, 1, 1, -1, -1])
alpha = solve_svm_dual(X, y)
print(f"Alphas: {alpha}")
print(f"Support vectors: {np.sum(alpha > 1e-4)}")
```

---

## 5. Decision Function

Once we have $\mathbf{w}^*$ and $b^*$:

$$
f(\mathbf{x}) = \text{sign}\left(\sum_{i=1}^n \alpha_i^* y_i \mathbf{x}_i^T \mathbf{x} + b^*\right)
$$

Only the support vectors (where $\alpha_i^* > 0$) contribute to this sum.

```python
def svm_predict(X_new, X_sv, y_sv, alpha_sv, b):
    """Predict using support vectors only."""
    scores = np.array([
        np.sum(alpha_sv * y_sv * (X_sv @ x)) + b 
        for x in X_new
    ])
    return np.sign(scores)
```

---

## 6. Geometric Interpretation of the Dual

The constraint $\sum \alpha_i y_i = 0$ ensures the hyperplane is balanced between classes. The term $\mathbf{x}_i^T\mathbf{x}_j$ measures similarity between data points — points that are similar (large dot product) and from different classes compete for large $\alpha$ values.

> **Check your intuition:** Why does the dual formulation only involve dot products $\mathbf{x}_i^T\mathbf{x}_j$ and not the raw data? *(Answer: The hyperplane orientation only depends on how data points relate to each other (similarities), not their absolute positions. This fact enables the kernel trick — we can replace dot products with kernel functions.)*
