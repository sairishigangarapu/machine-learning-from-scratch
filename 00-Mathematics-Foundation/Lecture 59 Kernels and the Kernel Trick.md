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

| Term | Definition | Significance |
|:---|:---|:---|
| $K(\mathbf{x}_i, \mathbf{x}_j)$ | Kernel function — similarity measure in feature space | Core of the kernel trick; replaces dot products $\mathbf{x}_i^T\mathbf{x}_j$ in the dual without computing $\phi$ explicitly |
| $\phi(\mathbf{x})$ | Feature map; transforms $\mathbf{x}$ to higher-dimensional space | May map to infinite dimensions (RBF kernel); never computed explicitly — only its dot products via $K$ |
| $\phi(\mathbf{x}_i)^T\phi(\mathbf{x}_j)$ | Dot product in feature space | Equivalent to $K(\mathbf{x}_i, \mathbf{x}_j)$, the kernel evaluation; avoids $O(D)$ computation in high-$D$ space |

we never need to compute $\phi(\mathbf{x})$ explicitly!

### Kernel Definition
A function $K(\mathbf{x}, \mathbf{z})$ is a valid kernel if the **Gram matrix** $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ is positive semi-definite for any set of data points.

### Mercer's Condition
$K$ is a valid kernel if and only if for any function $g(\mathbf{x})$ with finite norm:

$$
\int\int K(\mathbf{x}, \mathbf{z}) g(\mathbf{x}) g(\mathbf{z}) \, d\mathbf{x} \, d\mathbf{z} \ge 0
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $K(\mathbf{x}, \mathbf{z})$ | Candidate kernel function | Must be positive semi-definite to be a valid Mercer kernel |
| $g(\mathbf{x})$ | Any square-integrable function | Used in the integral condition to test kernel validity; condition must hold for all such $g$ |
| $\int\int \cdots \ge 0$ | Mercer's condition (necessary and sufficient) | Guarantees that $K$ corresponds to a dot product in some feature space — the mathematical foundation of the kernel trick |

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

| Term | Definition | Significance |
|:---|:---|:---|
| $K(\mathbf{x}, \mathbf{z})$ | Linear kernel | Simplest kernel; recovers standard linear SVM; no feature expansion |
| $\mathbf{x}^T\mathbf{z}$ | Standard dot product | Measures cosine similarity scaled by magnitudes; $O(d)$ per evaluation |

Equivalent to linear SVM. Use when $d$ is large.

### Polynomial Kernel
$$
K(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^T\mathbf{z} + c)^p
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $K(\mathbf{x}, \mathbf{z})$ | Polynomial kernel | Creates feature space of all monomials up to degree $p$; captures feature interactions |
| $p$ | Polynomial degree | Controls model complexity; $p=2$ gives quadratic boundary, $p=3$ cubic, etc.; higher $p$ risks overfitting |
| $c$ | Constant term | Controls influence of lower-degree terms; $c > 0$ includes all monomials of degree $\le p$ (not just degree $p$) |

* $p$ = degree, $c$ = constant
* Feature space dimension: $\binom{d+p}{p}$

### RBF (Gaussian) Kernel
$$
K(\mathbf{x}, \mathbf{z}) = \exp\left(-\gamma ||\mathbf{x} - \mathbf{z}||^2\right)
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $K(\mathbf{x}, \mathbf{z})$ | RBF (Gaussian) kernel | Most popular kernel; corresponds to infinite-dimensional feature space; universal approximator |
| $\gamma$ | Kernel bandwidth parameter | $\gamma = \frac{1}{2\sigma^2}$; large $\gamma$ = narrow kernel (local influence, may overfit); small $\gamma$ = wide kernel (smooth boundary) |
| $\exp(-\gamma ||\mathbf{x} - \mathbf{z}||^2)$ | Gaussian radial basis function | Decays smoothly with distance; always positive definite; $K(\mathbf{x}, \mathbf{x}) = 1$ (self-similarity is maximal) |

* $\gamma = \frac{1}{2\sigma^2}$ controls the "width"
* Feature space is **infinite-dimensional**
* Universal approximator

### Sigmoid Kernel
$$
K(\mathbf{x}, \mathbf{z}) = \tanh(\alpha \mathbf{x}^T\mathbf{z} + c)
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $K(\mathbf{x}, \mathbf{z})$ | Sigmoid kernel | Inspired by neural network activation functions; may not be PSD for all parameter choices |
| $\tanh(\cdot)$ | Hyperbolic tangent activation | Saturates at $\pm 1$; maps unbounded dot products into $(-1, 1)$ |
| $\alpha$ | Slope parameter | Scales input before activation; affects how quickly the kernel saturates |
| $c$ | Offset (bias) parameter | Shifts activation center; influences which dot products produce positive similarity |

**Caveat:** The sigmoid kernel is NOT always a valid (positive semi-definite) kernel. It only satisfies Mercer's condition for certain values of $\alpha$ and $c$. In practice, prefer the RBF kernel — it's always valid and generally performs better. Use sigmoid only when you have a specific reason (e.g., approximating a neural network).

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

| Term | Definition | Significance |
|:---|:---|:---|
| $\boldsymbol{\alpha}$ | Vector of dual variables (Lagrange multipliers) | Contains $n$ entries; most are zero (sparsity); only support vectors have $\alpha_i > 0$ |
| $\sum_i \alpha_i$ | Sum of dual variables | Linear term encouraging large margin; maximized in the dual objective |
| $-\frac{1}{2}\sum_i\sum_j \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$ | Quadratic penalty with kernel | Replaces $\mathbf{x}_i^T\mathbf{x}_j$ with $K(\mathbf{x}_i, \mathbf{x}_j)$ — this is how the kernel trick enters the optimization |
| $K(\mathbf{x}_i, \mathbf{x}_j)$ | Kernel evaluation between points $i$ and $j$ | Encodes similarity in the implicit feature space; no need to compute $\phi$ |

### Decision Function with Kernel
$$
f(\mathbf{x}) = \text{sign}\left(\sum_{i=1}^n \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right)
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $f(\mathbf{x})$ | Kernelized SVM prediction for new point $\mathbf{x}$ | Computes decision by comparing $\mathbf{x}$ to support vectors via the kernel function |
| $\sum_{i=1}^n \alpha_i y_i K(\mathbf{x}_i, \mathbf{x})$ | Weighted sum of kernel similarities to training points | Efficient: only support vectors ($\alpha_i > 0$) contribute; complexity is $O(n_s d)$ where $n_s$ is number of SVs |
| $K(\mathbf{x}_i, \mathbf{x})$ | Kernel between support vector $i$ and new point $\mathbf{x}$ | Measures how similar the new point is to each support vector in feature space |

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
