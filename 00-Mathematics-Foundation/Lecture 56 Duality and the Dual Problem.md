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
\min_{\mathbf{w}, b} \quad \frac{1}{2}||\mathbf{w}||^2
$$
$$
\text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1, \quad i = 1, \dots, n
$$

---

## 3. Constructing the Lagrangian

$$
\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}||\mathbf{w}||^2 - \sum_{i=1}^n \alpha_i \left[ y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1 \right]
$$

Rearranging:
$$
\mathcal{L} = \frac{1}{2}||\mathbf{w}||^2 - \mathbf{w}^T\left(\sum_i \alpha_i y_i \mathbf{x}_i\right) - b\left(\sum_i \alpha_i y_i\right) + \sum_i \alpha_i
$$

---

## 4. Deriving the Dual

### Step 1: Minimize over $\mathbf{w}$
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \mathbf{w} - \sum_i \alpha_i y_i \mathbf{x}_i = 0 \implies \mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i
$$

### Step 2: Minimize over $b$
$$
\frac{\partial \mathcal{L}}{\partial b} = -\sum_i \alpha_i y_i = 0 \implies \sum_i \alpha_i y_i = 0
$$

### Step 3: Substitute back
Plugging the expressions for $\mathbf{w}$ and the constraint on $b$ into $\mathcal{L}$:

$$
\mathcal{L}_D(\boldsymbol{\alpha}) = \sum_i \alpha_i - \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j
$$

### The Dual Problem
$$
\max_{\boldsymbol{\alpha}} \quad \sum_i \alpha_i - \frac{1}{2} \boldsymbol{\alpha}^T Q \boldsymbol{\alpha}
$$
$$
\text{s.t.} \quad \alpha_i \ge 0 \quad \forall i, \quad \sum_i \alpha_i y_i = 0
$$

where $Q_{ij} = y_i y_j \mathbf{x}_i^T\mathbf{x}_j$.

---

## 5. Complementary Slackness

The KKT complementary slackness condition states:

$$
\alpha_i \left[ y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1 \right] = 0 \quad \forall i
$$

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
