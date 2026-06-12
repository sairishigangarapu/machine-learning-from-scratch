## Kernels and the Kernel Trick

*Essential Mathematics for ML — Structured Notes*

---

## 1. The Limitation of Linear Classifiers

### Motivation and Intuition
A linear SVM can only draw straight decision boundaries. But what if the data is separable by a circle, a sine wave, or some complex curve? The **kernel trick** solves this by implicitly mapping data to a higher-dimensional space where a linear separator exists.

### The XOR Problem
No line can separate XOR. But mapping $(x_1, x_2) \to (x_1, x_2, x_1 x_2)$ makes it linearly separable.

```python
import numpy as np
from sklearn.svm import SVC

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([-1, 1, 1, -1])

# Linear SVM fails
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_xor, y_xor)
print(f"Linear accuracy: {svm_linear.score(X_xor, y_xor):.1f}")  # 0.5

# RBF kernel succeeds
svm_rbf = SVC(kernel='rbf', gamma=1)
svm_rbf.fit(X_xor, y_xor)
print(f"RBF accuracy: {svm_rbf.score(X_xor, y_xor):.1f}")  # 1.0
```

---

## 2. Feature Maps

### Explicit Feature Map
Map data to higher dimensions: $\phi: \mathbb{R}^d \to \mathbb{R}^D$ where $D \gg d$.

| Original | Feature Map | New Space |
|:---|:---|:---|
| $(x_1, x_2)$ | $(x_1, x_2, x_1^2, x_2^2, x_1 x_2)$ | $\mathbb{R}^5$ |
| $(x_1, x_2)$ | All monomials of degree $\le p$ | $\mathbb{R}^{\binom{d+p}{p}}$ |

**Problem:** Computing $\phi(\mathbf{x})$ explicitly is expensive when $D$ is very large (or infinite).

---

## 3. The Kernel Trick

### Key Insight
In the SVM dual, data only appears as dot products: $\mathbf{x}_i^T\mathbf{x}_j$. If we replace this with:

$$
K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T\phi(\mathbf{x}_j)
$$

we never need to compute $\phi(\mathbf{x})$ explicitly!

### Kernel Definition
A function $K(\mathbf{x}, \mathbf{z})$ is a valid kernel if the **Gram matrix** $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ is positive semi-definite for any set of data points.

### Mercer's Condition
$K$ is a valid kernel if and only if for any function $g(\mathbf{x})$ with finite norm:

$$
\int\int K(\mathbf{x}, \mathbf{z}) g(\mathbf{x}) g(\mathbf{z}) \, d\mathbf{x} \, d\mathbf{z} \ge 0
$$

```python
import numpy as np

def kernel_matrix(X, kernel_fn):
    """Compute the kernel (Gram) matrix."""
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_fn(X[i], X[j])
    return K

# Linear kernel: K(x, z) = x^T z
linear_kernel = lambda x, z: np.dot(x, z)

X = np.random.randn(5, 3)
K = kernel_matrix(X, linear_kernel)
print("Linear kernel matrix (should equal X @ X.T):")
print(np.allclose(K, X @ X.T))  # True
```

---

## 4. Common Kernels

### Linear Kernel
$$
K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T\mathbf{z}
$$
Equivalent to linear SVM. Use when $d$ is large.

### Polynomial Kernel
$$
K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T\mathbf{z} + c)^p
$$
* $p$ = degree, $c$ = constant
* Feature space dimension: $\binom{d+p}{p}$

### RBF (Gaussian) Kernel
$$
K(\mathbf{x}, \mathbf{z}) = \exp\left(-\gamma ||\mathbf{x} - \mathbf{z}||^2\right)
$$
* $\gamma = \frac{1}{2\sigma^2}$ controls the "width"
* Feature space is **infinite-dimensional**
* Universal approximator

### Sigmoid Kernel
$$
K(\mathbf{x}, \mathbf{z}) = \tanh(\alpha \mathbf{x}^T\mathbf{z} + c)
$$

```python
from sklearn.svm import SVC

kernels = {
    'linear': SVC(kernel='linear', C=1),
    'poly': SVC(kernel='poly', degree=3, C=1),
    'rbf': SVC(kernel='rbf', gamma='scale', C=1),
}

for name, model in kernels.items():
    model.fit(X_train, y_train)
    print(f"{name:8s}: accuracy = {model.score(X_test, y_test):.3f}")
```

---

## 5. The Kernel Trick in the Dual

### Dual with Kernel
$$
\max_{\boldsymbol{\alpha}} \quad \sum_i \alpha_i - \frac{1}{2}\sum_i\sum_j \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)
$$

### Decision Function with Kernel
$$
f(\mathbf{x}) = \text{sign}\left(\sum_{i=1}^n \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right)
$$

Only support vectors contribute to the sum.

```python
def kernel_decision(X_new, X_sv, y_sv, alpha_sv, b, kernel_fn):
    """SVM prediction using kernel function."""
    scores = np.zeros(len(X_new))
    for i, x in enumerate(X_new):
        scores[i] = np.sum(alpha_sv * y_sv * np.array([kernel_fn(x, x_sv) for x_sv in X_sv]))
    return np.sign(scores + b)
```

---

## 6. Hyperparameter Tuning

| Kernel | Hyperparameters | Typical Range |
|:---|:---|:---|
| **Polynomial** | $p$ (degree), $c$ (constant) | $p \in \{2, 3, 4\}$, $c \in \{0, 1\}$ |
| **RBF** | $\gamma$ (bandwidth), $C$ (regularization) | $\gamma \in [0.001, 100]$, $C \in [0.1, 1000]$ |
| **Linear** | $C$ | $C \in [0.01, 100]$ |

**ML Connection:** Use grid search or random search to tune hyperparameters. Cross-validation ensures the model generalizes to unseen data.

---

## 7. Computational Complexity

| Method | Training | Prediction |
|:---|:---|:---|
| **Explicit $\phi$** | $O(Dn^2)$ to $O(Dn^3)$ | $O(Dd)$ |
| **Kernel trick** | $O(n^2 d)$ to $O(n^3)$ | $O(n_s d)$ where $n_s$ = # SVs |

The kernel trick is more efficient when the implicit feature space dimension $D$ is very large (e.g., infinite for RBF), but the number of support vectors $n_s$ is small.

> **Check your intuition:** Can you use any function $K(\mathbf{x}, \mathbf{z})$ as a kernel? *(Answer: No. $K$ must be positive semi-definite (Mercer's condition). For example, $K(\mathbf{x}, \mathbf{z}) = -||\mathbf{x} - \mathbf{z}||^2$ is not a valid kernel because it's not positive semi-definite.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 58: Soft Margin SVM](Lecture%2058%20Soft%20Margin%20SVM.md) — Handles non-separable data
- **Next:** [Lecture 60: SVM Implementation in Python](Lecture%2060%20SVM%20Implementation%20in%20Python.md) — Practical implementation with kernels
- **Related:** [Lecture 54: Introduction to Support Vector Machines](Lecture%2054%20Introduction%20to%20Support%20Vector%20Machines.md) — SVM fundamentals
- **Related:** [Lecture 58: Soft Margin SVM](Lecture%2058%20Soft%20Margin%20SVM.md) — Combines with kernel trick
- **Related:** [Lecture 60: SVM Implementation in Python](Lecture%2060%20SVM%20Implementation%20in%20Python.md) — Practical application of kernels
