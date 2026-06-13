## Hard Margin SVM

*Essential Mathematics for ML — Structured Notes*

---

## 1. Primal Formulation (Complete)

### Motivation and Intuition
The hard margin SVM requires all training points to be correctly classified with a margin of at least 1. This is the original SVM formulation — simple, elegant, and powerful when data is linearly separable.

### Mathematical Formulation
$$
\begin{aligned}
\min_{\mathbf{w}, b} \quad &\frac{1}{2}||\mathbf{w}||^2 \\
\text{s.t.} \quad &y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1, \quad i = 1, \dots, n
\end{aligned}
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\frac{1}{2}||\mathbf{w}||^2$ | Convex objective | Minimizing $||\mathbf{w}||^2$ is equivalent to maximizing the margin $\frac{2}{||\mathbf{w}||}$ |
| $y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1$ | Hard margin constraints | Ensure correct classification with distance $\ge 1/||\mathbf{w}||$ from boundary |

**Note:** The constraint $y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1$ ensures:
1. Correct classification ($y_i(\mathbf{w}^T\mathbf{x}_i + b) > 0$)
2. Minimum distance of $\frac{1}{||\mathbf{w}||}$ from the boundary

---

## 2. Dual Formulation (Complete)

$$
\begin{aligned}
\max_{\boldsymbol{\alpha}} \quad &\sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j \\
\text{s.t.} \quad &\alpha_i \ge 0 \quad \forall i, \quad \sum_{i=1}^n \alpha_i y_i = 0
\end{aligned}
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\boldsymbol{\alpha}$ | Lagrange multipliers | $\alpha_i \ge 0$; non-zero only for support vectors — sparsity property |
| $\sum_{i=1}^n \alpha_i$ | Linear term | Maximizing this pushes the margin wider |
| $\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j$ | Quadratic term | Pairwise interactions; only depends on dot products — enables the kernel trick |
| $\sum_{i=1}^n \alpha_i y_i = 0$ | Balance constraint | Ensures the hyperplane is centered between classes |

### Solution Recovery
$$
\begin{aligned}
\mathbf{w}^* &= \sum_{i=1}^n \alpha_i^* y_i \mathbf{x}_i \\
b^* &= y_s - \mathbf{w}^{*T}\mathbf{x}_s \quad \text{for any support vector } \mathbf{x}_s
\end{aligned}
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\mathbf{w}^*$ | Optimal weight vector | Weighted combination of support vectors; $\mathbf{w}^* = \sum \alpha_i^* y_i \mathbf{x}_i$ |
| $b^*$ | Optimal bias | Computed from any support vector; $b^* = y_s - \mathbf{w}^{*T}\mathbf{x}_s$ |
| $\alpha_i^*$ | Optimal dual variables | Only support vectors ($\alpha_i^* > 0$) contribute to $\mathbf{w}^*$ and the decision function |

---

## 3. Complete Implementation

```python
import numpy as np
from cvxopt import matrix, solvers

def hard_margin_svm(X, y):
    """
    Solve hard margin SVM using cvxopt.
    
    Parameters:
        X: (n, d) feature matrix
        y: (n,) labels in {-1, +1}
    
    Returns:
        w: weight vector
        b: bias
        alpha: Lagrange multipliers
    """
    n, d = X.shape
    solvers.options['show_progress'] = False
    
    # Build QP: minimize (1/2)alpha^T Q alpha - 1^T alpha
    # Q_ij = y_i y_j x_i^T x_j
    Q = np.outer(y, y) * (X @ X.T)
    
    P = matrix(Q, tc='d')
    q = matrix(-np.ones(n), tc='d')
    
    # Constraints: alpha_i >= 0
    G = matrix(-np.eye(n), tc='d')
    h = matrix(np.zeros(n), tc='d')
    
    # Equality: sum(alpha_i * y_i) = 0
    A = matrix(y.reshape(1, -1).astype(float), tc='d')
    b_eq = matrix(0.0, tc='d')
    
    sol = solvers.qp(P, q, G, h, A, b_eq)
    alpha = np.array(sol['x']).flatten()
    
    # Recover w and b
    w = np.sum(alpha[:, None] * y[:, None] * X, axis=0)
    
    # Find support vector for bias computation
    sv_indices = np.where(alpha > 1e-5)[0]
    b = y[sv_indices[0]] - X[sv_indices[0]] @ w
    
    return w, b, alpha

# Example
X = np.array([[2, 3], [1, 1], [3, 3], [6, 5], [7, 7], [8, 6]])
y = np.array([1, 1, 1, -1, -1, -1])

w, b, alpha = hard_margin_svm(X, y)
print(f"w = {w}")
print(f"b = {b:.4f}")
print(f"Number of support vectors: {np.sum(alpha > 1e-5)}")
```

---

## 4. Decision Function

$$
f(\mathbf{x}) = \text{sign}(\mathbf{w}^{*T}\mathbf{x} + b^*)
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $f(\mathbf{x})$ | Hard margin SVM prediction for new point $\mathbf{x}$ | Returns $\pm 1$ using only the weight vector and bias from the primal solution |
| $\mathbf{w}^*$ | Optimal weight vector from primal training | Defines the orientation of the maximum-margin hyperplane |
| $b^*$ | Optimal bias from primal training | Determines the offset of the decision boundary from the origin |

```python
def predict(X_new, w, b):
    return np.sign(X_new @ w + b)

print(f"Predictions: {predict(X, w, b)}")
print(f"True labels: {y}")
```

---

## 5. Margin Analysis

### Margin Width
$$
\text{Margin} = \frac{2}{||\mathbf{w}^*||}
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\text{Margin}$ | Total width of the margin region | The distance between the two margin boundaries $\mathbf{w}^T\mathbf{x} + b = \pm 1$ |
| $\mathbf{w}^*$ | Optimal weight vector | Minimizing $||\mathbf{w}^*||$ maximizes this margin — the core optimization goal |
| $\frac{2}{||\mathbf{w}^*||}$ | Margin formula | Directly links the optimization objective to the geometric quantity we want to maximize |

### Perpendicular Distance
The distance from any point $\mathbf{x}_0$ to the decision boundary:

$$
d = \frac{|\mathbf{w}^{*T}\mathbf{x}_0 + b^*|}{||\mathbf{w}^*||}
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $d$ | Perpendicular distance from point $\mathbf{x}_0$ to the decision boundary | Measures how far a point is from the separating hyperplane in Euclidean space |
| $\mathbf{x}_0$ | Any data point (could be training or test) | Used to compute distances for margin analysis and to check which points are support vectors |
| $|\mathbf{w}^{*T}\mathbf{x}_0 + b^*|$ | Absolute decision score at $\mathbf{x}_0$ | Numerator is the functional margin's magnitude; dividing by $||\mathbf{w}^*||$ makes it scale-invariant |

```python
margin_width = 2 / np.linalg.norm(w)
print(f"Margin width: {margin_width:.4f}")

# Distances of all points to boundary
distances = np.abs(X @ w + b) / np.linalg.norm(w)
print(f"Distances: {distances}")
print(f"Min distance (should be ~margin/2): {np.min(distances):.4f}")
```

---

## 6. Limitations of Hard Margin SVM

| Limitation | Consequence |
|:---|:---|
| Requires linear separability | Fails if data overlaps |
| Sensitive to outliers | One outlier can drastically reduce margin |
| No probabilistic output | Only provides classification, not confidence |

**ML Connection:** These limitations motivate **soft margin SVM** (Lecture 58), which allows some violations of the margin constraint. Most real-world datasets are not perfectly linearly separable, making soft margin SVM the practical choice.

> **Check your intuition:** If you add a single outlier far from the existing data, how does the hard margin SVM change? *(Answer: The margin shrinks dramatically to accommodate the outlier. The entire decision boundary may rotate. This is why hard margin SVM is rarely used in practice — soft margin SVM handles outliers gracefully.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 56: Duality and the Dual Problem](Lecture%2056%20Duality%20and%20the%20Dual%20Problem.md) — Theoretical foundation for dual formulation
- **Next:** [Lecture 58: Soft Margin SVM](Lecture%2058%20Soft%20Margin%20SVM.md) — Handles non-separable data
- **Related:** [Lecture 55: Maximum Margin Classification](Lecture%2055%20Maximum%20Margin%20Classification.md) — Introduces maximum margin concept
- **Related:** [Lecture 56: Duality and the Dual Problem](Lecture%2056%20Duality%20and%20the%20Dual%20Problem.md) — Dual derivation
- **Related:** [Lecture 58: Soft Margin SVM](Lecture%2058%20Soft%20Margin%20SVM.md) — Practical extension for real-world data
