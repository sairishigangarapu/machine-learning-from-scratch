## Maximum Margin Classification

*Essential Mathematics for ML — Structured Notes*

---

## 1. Formal Problem Setup

### Motivation and Intuition
We want to find the hyperplane that separates two classes with maximum margin. This is a constrained optimization problem — minimize the norm of the weight vector subject to all points being correctly classified with a minimum functional margin of 1.

### The Primal Problem
Given training data $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ where $y_i \in \{-1, +1\}$:

$$
\begin{aligned}
\min_{\mathbf{w}, b} \quad &\frac{1}{2}||\mathbf{w}||^2 \\
\text{s.t.} \quad &y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1 \quad \forall i = 1, \dots, n
\end{aligned}
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\frac{1}{2}||\mathbf{w}||^2$ | Objective — half squared norm of weight vector | Minimizing this maximizes the margin; convex quadratic — guarantees global optimum |
| $y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1$ | Margin constraint | Forces all points to be correctly classified with functional margin at least 1 |
| $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$ | Training dataset | $n$ labeled examples; $y_i \in \{-1, +1\}$ encodes class membership |

---

## 2. Lagrangian Formulation

### Lagrangian
$$
\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}||\mathbf{w}||^2 - \sum_{i=1}^n \alpha_i \left[ y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1 \right]
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha})$ | Lagrangian function | Encodes both the objective and constraints into a single unconstrained function; enables dual derivation |
| $\frac{1}{2}||\mathbf{w}||^2$ | Regularization term (half squared norm of weights) | Minimizing this maximizes the margin; convex and differentiable — easy to optimize |
| $\alpha_i$ | Lagrange multiplier for constraint $i$ | $\alpha_i \ge 0$; only support vectors have $\alpha_i > 0$; controls how hard each constraint is enforced |
| $y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1$ | Slack form of the margin constraint | At optimum, $= 0$ for support vectors, $> 0$ for points outside margin |

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
\begin{aligned}
\max_{\boldsymbol{\alpha}} \quad &\sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j \\
\text{s.t.} \quad &\alpha_i \ge 0 \quad \forall i, \quad \sum_{i=1}^n \alpha_i y_i = 0
\end{aligned}
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\boldsymbol{\alpha}$ | Vector of Lagrange multipliers | Dual variables; $\alpha_i \ge 0$; only support vectors have $\alpha_i > 0$ |
| $\sum_{i=1}^n \alpha_i$ | Linear term in dual objective | Encourages large margin; maximized in the dual |
| $-\frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j$ | Quadratic penalty term | Captures pairwise interactions between points; depends only on dot products |
| $\sum_{i=1}^n \alpha_i y_i = 0$ | Equality constraint | Balances total weight of positive and negative support vectors |

### Why the Dual?
* Only depends on dot products $\mathbf{x}_i^T\mathbf{x}_j$ — enables the kernel trick
* Number of variables equals number of data points (not dimensionality)
* Sparsity: most $\alpha_i = 0$

### Solution Recovery
Once $\boldsymbol{\alpha}^*$ is found:
$$
\begin{aligned}
\mathbf{w}^* &= \sum_{i=1}^n \alpha_i^* y_i \mathbf{x}_i \\
b^* &= y_s - \mathbf{w}^{*T}\mathbf{x}_s \quad \text{(for any support vector $\mathbf{x}_s$)}
\end{aligned}
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\mathbf{w}^*$ | Optimal weight vector | Expressed as weighted sum of support vectors — sparse representation |
| $\alpha_i^*$ | Optimal Lagrange multiplier for point $i$ | Non-zero only for support vectors; measures influence of each point |
| $b^*$ | Optimal bias | Computed from any support vector on the margin; ensures decision boundary is centered |
| $\mathbf{x}_s$ | A support vector (point on the margin) | Any point with $\alpha_s > 0$ can be used to compute $b^*$ |

---

## 4. Quadratic Programming

The dual is a **Quadratic Program (QP)**:

$$
\begin{aligned}
\min_{\boldsymbol{\alpha}} \quad &\frac{1}{2}\boldsymbol{\alpha}^T Q \boldsymbol{\alpha} - \mathbf{1}^T\boldsymbol{\alpha} \\
\text{s.t.} \quad &\mathbf{y}^T\boldsymbol{\alpha} = 0, \quad \boldsymbol{\alpha} \ge 0
\end{aligned}
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $Q$ | Gram matrix with $Q_{ij} = y_i y_j \mathbf{x}_i^T \mathbf{x}_j$ | Encodes all pairwise similarities scaled by label agreement; must be positive semi-definite |
| $\mathbf{1}$ | Vector of all ones | Used in linear term $\mathbf{1}^T\boldsymbol{\alpha}$; equivalently $\sum \alpha_i$ |
| $\mathbf{y}^T\boldsymbol{\alpha} = 0$ | Equality constraint in QP form | Compact representation of $\sum \alpha_i y_i = 0$ |
| $\boldsymbol{\alpha} \ge 0$ | Non-negativity constraint | Ensures dual feasibility |

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

| Term | Definition | Significance |
|:---|:---|:---|
| $f(\mathbf{x})$ | Decision function for a new point $\mathbf{x}$ | Outputs predicted class ($\pm 1$) based on support vectors only |
| $\alpha_i^*$ | Optimal Lagrange multiplier for point $i$ | Only non-zero for support vectors; makes prediction sparse — only a few training points matter |
| $\mathbf{x}_i^T \mathbf{x}$ | Dot product between support vector and new point | Measures similarity; this is what the kernel trick replaces with $K(\mathbf{x}_i, \mathbf{x})$ |
| $b^*$ | Optimal bias term | Computed from any support vector on the margin; shifts decision threshold |

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

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 54: Introduction to Support Vector Machines](Lecture%2054%20Introduction%20to%20Support%20Vector%20Machines.md) — Conceptual introduction to SVMs
- **Next:** [Lecture 56: Duality and the Dual Problem](Lecture%2056%20Duality%20and%20the%20Dual%20Problem.md) — Theoretical foundation for dual formulation
- **Related:** [Lecture 42: Constrained Optimization-I](Lecture%2042%20Constrained%20Optimization-I.md) — Lagrangian and KKT conditions
- **Related:** [Lecture 56: Duality and the Dual Problem](Lecture%2056%20Duality%20and%20the%20Dual%20Problem.md) — Detailed derivation of dual
- **Related:** [Lecture 57: Hard Margin SVM](Lecture%2057%20Hard%20Margin%20SVM.md) — Complete implementation of hard margin
