## Duality and the Dual Problem

*Essential Mathematics for ML — Structured Notes*

---

## 1. Why Duality?

### Motivation and Intuition
In optimization, we often have a **primal** problem with constraints. Duality transforms it into a **dual** problem that is often easier to solve. For SVMs, the dual formulation reveals the role of support vectors and enables the kernel trick.

### Strong vs Weak Duality

| Property | Meaning |
|:---|:---|
| **Weak duality** | Dual optimal value ≤ Primal optimal value (always holds) |
| **Strong duality** | Dual optimal = Primal optimal (holds when Slater's condition is satisfied) |

**Slater's condition:** If there exists a strictly feasible point (all constraints satisfied with strict inequality), then strong duality holds. For SVMs, if the data is linearly separable, strong duality holds.

---

## 2. Primal Formulation (Review)

$$
\begin{aligned}
\min_{\mathbf{w}, b} \quad &\frac{1}{2}||\mathbf{w}||^2 \\
\text{s.t.} \quad &y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1, \quad i = 1, \dots, n
\end{aligned}
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\frac{1}{2}||\mathbf{w}||^2$ | Regularization term | Maximizes margin; convex quadratic — always has a unique global minimum |
| $y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1$ | Hard margin constraints | Must hold for all $i$; define the feasible region of the primal problem |

---

## 3. Constructing the Lagrangian

$$
\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}||\mathbf{w}||^2 - \sum_{i=1}^n \alpha_i \left[ y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1 \right]
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\mathcal{L}$ | Lagrangian; combines objective and constraints | Foundation for deriving the dual problem; saddle point gives primal and dual optimal |
| $\frac{1}{2}||\mathbf{w}||^2$ | Regularization (margin maximizer) | Convex quadratic; ensures unique global optimum |
| $\alpha_i \ge 0$ | Lagrange multipliers | Enforce the inequality constraints; only non-zero at optimal solution for active constraints (support vectors) |
| $y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1$ | Margin violation (should be $\ge 0$) | At optimum, equals $0$ for points exactly on margin boundary |

Rearranging:
$$
\mathcal{L} = \frac{1}{2}||\mathbf{w}||^2 - \mathbf{w}^T\left(\sum_i \alpha_i y_i \mathbf{x}_i\right) - b\left(\sum_i \alpha_i y_i\right) + \sum_i \alpha_i
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\mathbf{w}^T(\sum_i \alpha_i y_i \mathbf{x}_i)$ | Linear term from expanding Lagrangian | Shows that $\mathbf{w}$ interacts with data only through $\sum \alpha_i y_i \mathbf{x}_i$ — a weighted combination of training points |
| $b(\sum_i \alpha_i y_i)$ | Bias term from expansion | Setting the gradient w.r.t $b$ to zero gives $\sum \alpha_i y_i = 0$, a key dual constraint |
| $\sum_i \alpha_i$ | Sum of Lagrange multipliers | This is what the dual maximizes — it appears from the $+1$ in the constraint $y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1$ |

---

## 4. Deriving the Dual

### Step 1: Minimize over $\mathbf{w}$
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \mathbf{w} - \sum_i \alpha_i y_i \mathbf{x}_i = 0 \implies \mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\frac{\partial \mathcal{L}}{\partial \mathbf{w}}$ | Gradient of Lagrangian w.r.t weights | Setting to zero gives the stationarity KKT condition |
| $\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$ | Optimal weight vector as linear combination of data | Key result: $\mathbf{w}$ lives in the span of training points; only points with $\alpha_i > 0$ (support vectors) contribute |
| $\sum_i \alpha_i y_i \mathbf{x}_i$ | Weighted sum of training points | Each point contributes proportionally to its Lagrange multiplier and label |

### Step 2: Minimize over $b$
$$
\frac{\partial \mathcal{L}}{\partial b} = -\sum_i \alpha_i y_i = 0 \implies \sum_i \alpha_i y_i = 0
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\frac{\partial \mathcal{L}}{\partial b}$ | Gradient of Lagrangian w.r.t bias | Setting to zero yields the constraint that the weighted sum of labels must be zero |
| $\sum_i \alpha_i y_i = 0$ | Dual feasibility constraint | Balances the total weight of positive and negative support vectors: $\sum_{y_i=+1} \alpha_i = \sum_{y_i=-1} \alpha_i$ |

### Step 3: Substitute back
Plugging the expressions for $\mathbf{w}$ and the constraint on $b$ into $\mathcal{L}$:

$$
\mathcal{L}_D(\boldsymbol{\alpha}) = \sum_i \alpha_i - \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\mathcal{L}_D(\boldsymbol{\alpha})$ | Dual Lagrangian (depends only on $\boldsymbol{\alpha}$) | Eliminates $\mathbf{w}$ and $b$; maximized to find the optimal SVM solution |
| $\sum_i \alpha_i$ | Linear term — reward for maximizing margin | Larger margin = larger $\alpha_i$ sum; encourages as many points as possible to be correctly classified |
| $-\frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j$ | Quadratic penalty — pairwise interactions | Points from different classes ($y_i y_j = -1$) with high similarity ($\mathbf{x}_i^T\mathbf{x}_j$ large) compete, limiting $\alpha$ values |

### The Dual Problem
$$
\begin{aligned}
\max_{\boldsymbol{\alpha}} \quad &\sum_i \alpha_i - \frac{1}{2} \boldsymbol{\alpha}^T Q \boldsymbol{\alpha} \\
\text{s.t.} \quad &\alpha_i \ge 0 \quad \forall i, \quad \sum_i \alpha_i y_i = 0
\end{aligned}
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\boldsymbol{\alpha}$ | Vector of dual variables | $n$ Lagrange multipliers, one per margin constraint |
| $\boldsymbol{\alpha}^T Q \boldsymbol{\alpha}$ | Quadratic form in the dual objective | $Q_{ij} = y_i y_j \mathbf{x}_i^T\mathbf{x}_j$; captures label-adjusted similarities between all pairs |
| $\alpha_i \ge 0$ | Dual feasibility (inequality) | Lagrange multipliers for margin constraints must be non-negative |
| $\sum_i \alpha_i y_i = 0$ | Dual feasibility (equality) | Arises from gradient w.r.t $b$; balances positive/negative support vector weights |

where $Q_{ij} = y_i y_j \mathbf{x}_i^T\mathbf{x}_j$.

---

## 5. Complementary Slackness

The KKT complementary slackness condition states:

$$
\alpha_i \left[ y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1 \right] = 0 \quad \forall i
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\alpha_i [y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1] = 0$ | Complementary slackness (KKT condition) | Forces $\alpha_i = 0$ OR the point to be exactly on the margin — never both non-zero |
| $\alpha_i$ | Lagrange multiplier for point $i$ | $\alpha_i > 0$ only when $y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1$ — the point is a support vector |
| $y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1$ | Margin slack | $> 0$ means point is outside margin; then $\alpha_i$ must be $0$ (point does not affect solution) |

This means for each data point, either:
1. $\alpha_i = 0$ — the point is not a support vector, OR
2. $y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1$ — the point lies exactly on the margin

```python
import numpy as np

def analyze_kkt(X, y, w, b, alpha, tol=1e-3):
    """Check KKT conditions for SVM solution."""
    n = len(y)
    margins = y * (X @ w + b)
    
    for i in range(n):
        if alpha[i] > tol:
            # Support vector
            assert abs(margins[i] - 1.0) < tol, f"SV {i} not on margin"
        else:
            # Non-support vector
            assert margins[i] > 1.0 - tol, f"Non-SV {i} inside margin"
    
    print("All KKT conditions satisfied!")
```

---

## 6. Why the Dual is Useful for SVMs

### Kernel Trick
The dual only involves dot products $\mathbf{x}_i^T\mathbf{x}_j$. We can replace these with kernel evaluations $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T\phi(\mathbf{x}_j)$ to implicitly map to higher dimensions.

### Sparsity
Most $\alpha_i = 0$ (non-support vectors). The solution is sparse in the dual, making it memory-efficient.

### Dimensionality
If $d \gg n$ (high dimensions, few samples), the dual has fewer variables than the primal.

---

## 7. Geometric Interpretation

The constraint $\sum \alpha_i y_i = 0$ can be rewritten as $\sum_{i: y_i=1} \alpha_i = \sum_{i: y_i=-1} \alpha_i$. This means the total "weight" assigned to positive class support vectors equals that of negative class support vectors.

> **Check your intuition:** If the data is NOT linearly separable, does strong duality still hold? *(Answer: No. Slater's condition requires strict feasibility, which fails when no separating hyperplane exists. This is why we need soft margin SVMs — they reformulate the problem to restore strong duality.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 55: Maximum Margin Classification](Lecture%2055%20Maximum%20Margin%20Classification.md) — Primal formulation of SVM
- **Next:** [Lecture 57: Hard Margin SVM](Lecture%2057%20Hard%20Margin%20SVM.md) — Complete hard margin implementation
- **Related:** [Lecture 42: Constrained Optimization-I](Lecture%2042%20Constrained%20Optimization-I.md) — Theoretical foundation for duality
- **Related:** [Lecture 55: Maximum Margin Classification](Lecture%2055%20Maximum%20Margin%20Classification.md) — Introduces dual formulation
- **Related:** [Lecture 58: Soft Margin SVM](Lecture%2058%20Soft%20Margin%20SVM.md) — Extends dual to handle non-separable data
