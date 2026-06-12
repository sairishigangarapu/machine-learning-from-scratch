## Soft Margin SVM

*Essential Mathematics for ML — Structured Notes*

---

## 1. Motivation for Soft Margin

### Why Hard Margin Fails
Real-world data is rarely perfectly linearly separable. Even if it is, outliers can drastically shrink the margin. Hard margin SVM is too rigid.

### The Solution: Allow Violations
Soft margin SVM introduces **slack variables** $\xi_i \ge 0$ that measure how much each point violates the margin constraint.

---

## 2. Primal Formulation

$$
\begin{aligned}
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \quad &\frac{1}{2}||\mathbf{w}||^2 + C \sum_{i=1}^n \xi_i \\
\text{s.t.} \quad &y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0
\end{aligned}
$$

**Parameter $C$ controls the trade-off:**

| $C$ value | Behavior |
|:---|:---|
| Large $C$ | Few margin violations (approaches hard margin) |
| Small $C$ | More violations allowed (wider margin) |
| $C \to \infty$ | Hard margin SVM |
| $C \to 0$ | Maximum margin, ignoring misclassifications |

### Hinge Loss Interpretation
The soft margin objective is equivalent to minimizing **hinge loss** + regularization:

$$
\min_{\mathbf{w}, b} \quad \sum_{i=1}^n \max(0, 1 - y_i(\mathbf{w}^T\mathbf{x}_i + b)) + \frac{\lambda}{2}||\mathbf{w}||^2
$$

where $\lambda = \frac{1}{C}$.

```python
import numpy as np

def hinge_loss(y_true, y_pred, w, b, C):
    """Compute soft margin SVM objective."""
    margins = y_true * (y_pred + b)
    loss = np.sum(np.maximum(0, 1 - margins))
    reg = 0.5 * np.sum(w**2)
    return reg + C * loss
```

---

## 3. Dual Formulation

$$
\begin{aligned}
\max_{\boldsymbol{\alpha}} \quad &\sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j \\
\text{s.t.} \quad &0 \le \alpha_i \le C \quad \forall i, \quad \sum_{i=1}^n \alpha_i y_i = 0
\end{aligned}
$$

The only change from hard margin: $\alpha_i \le C$ (upper bound).

### Support Vector Classification

| $\alpha_i$ value | Point position | Classification |
|:---|:---|:---|
| $\alpha_i = 0$ | Correctly classified, outside margin | Non-support vector |
| $0 < \alpha_i < C$ | On the margin boundary | Support vector |
| $\alpha_i = C$ | Inside margin or misclassified | Bounded support vector |

```python
import numpy as np
from cvxopt import matrix, solvers

def soft_margin_svm(X, y, C=1.0):
    """Solve soft margin SVM."""
    n = X.shape[0]
    solvers.options['show_progress'] = False
    
    Q = np.outer(y, y) * (X @ X.T)
    P = matrix(Q, tc='d')
    q = matrix(-np.ones(n), tc='d')
    
    # 0 <= alpha_i <= C
    G = matrix(np.vstack([-np.eye(n), np.eye(n)]), tc='d')
    h = matrix(np.concatenate([np.zeros(n), C * np.ones(n)]), tc='d')
    
    A = matrix(y.reshape(1, -1).astype(float), tc='d')
    b_eq = matrix(0.0, tc='d')
    
    sol = solvers.qp(P, q, G, h, A, b_eq)
    alpha = np.array(sol['x']).flatten()
    
    # Recover w
    w = np.sum(alpha[:, None] * y[:, None] * X, axis=0)
    
    # Find support vectors (on margin: 0 < alpha < C)
    sv_mask = (alpha > 1e-5) & (alpha < C - 1e-5)
    if np.any(sv_mask):
        b = y[sv_mask][0] - X[sv_mask][0] @ w
    else:
        # Fallback: use any support vector
        sv_mask = alpha > 1e-5
        b = y[sv_mask][0] - X[sv_mask][0] @ w
    
    return w, b, alpha
```

---

## 4. Complete Example

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Non-linearly separable data
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
y = np.where(y == 0, -1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Compare different C values
for C in [0.01, 0.1, 1, 10, 100]:
    svm = SVC(kernel='linear', C=C)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    n_sv = svm.n_support_.sum()
    print(f"C={C:6.2f} | Accuracy: {acc:.3f} | Support vectors: {n_sv}")
```

---

## 5. Slack Variable Interpretation

Each $\xi_i$ measures the distance of point $i$ from its correct margin boundary:

| $\xi_i$ value | Meaning |
|:---|:---|
| $\xi_i = 0$ | Correctly classified, on or outside margin |
| $0 < \xi_i < 1$ | Correctly classified, inside margin |
| $\xi_i = 1$ | On the decision boundary |
| $\xi_i > 1$ | Misclassified |

**Total margin violations:** $\sum \xi_i$ measures total "error" of the model.

---

## 6. Comparison with Other Classifiers

| Method | Objective | Kernel | Robustness |
|:---|:---|:---|:---|
| **Hard Margin SVM** | Min $||\mathbf{w}||^2$ s.t. $y_i f(\mathbf{x}_i) \ge 1$ | Yes | Low |
| **Soft Margin SVM** | Min $||\mathbf{w}||^2 + C\sum\xi_i$ | Yes | High |
| **Logistic Regression** | Min $\sum \log(1 + e^{-y_i f(\mathbf{x}_i)}) + \lambda||\mathbf{w}||^2$ | Yes | High |
| **Perceptron** | Min $\sum \max(0, -y_i f(\mathbf{x}_i))$ | No | Low |

> **Check your intuition:** If $C$ is very large, does the soft margin SVM always find the hard margin solution? *(Answer: Only if the data is linearly separable. If it's not, even large $C$ cannot satisfy all constraints, and some $\xi_i$ will be positive. The model will be close to hard margin but with unavoidable violations.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 57: Hard Margin SVM](Lecture%2057%20Hard%20Margin%20SVM.md) — Idealized case for separable data
- **Next:** [Lecture 59: Kernels and the Kernel Trick](Lecture%2059%20Kernels%20and%20the%20Kernel%20Trick.md) — Non-linear extensions
- **Related:** [Lecture 54: Introduction to Support Vector Machines](Lecture%2054%20Introduction%20to%20Support%20Vector%20Machines.md) — SVM fundamentals
- **Related:** [Lecture 57: Hard Margin SVM](Lecture%2057%20Hard%20Margin%20SVM.md) — Theoretical hard margin case
- **Related:** [Lecture 59: Kernels and the Kernel Trick](Lecture%2059%20Kernels%20and%20the%20Kernel%20Trick.md) — Combines soft margin with kernels
