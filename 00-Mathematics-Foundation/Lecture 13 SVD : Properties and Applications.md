## Singular Value Decomposition: Properties and Matrix Norms

*Essential Mathematics for ML — Advanced Structured Notes*

---

## 1. Geometric Interpretation of SVD

SVD reveals that any linear transformation $A = U \Sigma V^T$ is a sequence of three distinct geometric operations:

1. **First Rotation ($V^T$):** The input space (a unit circle) is rotated such that its principal axes align with the coordinate system.
2. **Scaling ($\Sigma$):** The rotated space is stretched or squashed by the singular values $\sigma_i$. This transforms the circle into an ellipse.
3. **Second Rotation ($U$):** The ellipse is rotated again to its final orientation in the output space.

---

## 2. Fundamental Subspace Properties

SVD doesn't just factor a matrix; it completely decrypts it. SVD isolates the four fundamental topological subspaces of an $m \times n$ matrix $A$ (where $\text{rank}(A) = r$) perfectly inside the columns of $U$ and $V$.

| Subspace | Basis (Where to find it) | Dimension |
| --- | --- | --- |
| **Range Space** $\text{Col}(A)$ | First $r$ columns of $U$ | $r$ |
| **Null Space** $\text{Null}(A)$ | Last $n-r$ columns of $V$ | $n-r$ |
| **Row Space** $\text{Row}(A)$ | First $r$ columns of $V$ | $r$ |
| **Left Null Space** $\text{Null}(A^T)$ | Last $m-r$ columns of $U$ | $m-r$ |

> **Machine Learning Core Concept:** The columns of $V$ tied to zero (or infinitesimally small) singular values form the **Null Space**. These represent the specific features in your dataset that the mathematical transformation absolutely ignores or mathematically deletes.

---

## 3. The Moore-Penrose Pseudo-Inverse ($A^+$)

### Motivation
In standard Linear Regression, you want to solve $\mathbf{w} = X^{-1} \mathbf{y}$. But $X$ is never uniformly square, rendering standard matrix inversion $X^{-1}$ mathematically illegal.

To solve this, we engineer a **Pseudo-Inverse** $A^+$. It acts as the "best possible mathematical compromise" when a true inverse doesn't exist.

$$
A^+ = V \Sigma^+ U^T
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $A^+$ | The Moore-Penrose pseudo-inverse of $A$ | Generalizes matrix inversion to non-square and singular matrices |
| $V$ | Right singular vectors from SVD of $A$ | Provides the orthogonal basis for the row space |
| $\Sigma^+$ | Pseudo-inverse of the diagonal singular value matrix | Inverts non-zero singular values, zeros remain zero |
| $U^T$ | Transpose of left singular vectors | Provides the orthogonal basis for the column space |

To build $\Sigma^+$, we transpose $\Sigma$ and take the strict reciprocal of all non-zero singular values ($1/\sigma_i$). Critically, if $\sigma_i = 0$, it remains $0$ (preventing catastrophic division errors).

### Worked Example: Pseudo-Inverse

**Given:** 

$$
A = \begin{bmatrix} 4 & 11 & 14 \\ 8 & 7 & -2 \end{bmatrix} \text{ (Rank 2)}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $A$ | Example matrix for pseudo-inverse computation | $2 \times 3$ rank-2 matrix used to demonstrate the Moore-Penrose pseudo-inverse |
| $4, 11, 14, \dots$ | Entries of $A$ | Coefficients of the linear transformation |

1. **Identify Components:** $U$ is $2 \times 2$, $\Sigma$ is $2 \times 3$, and $V$ is $3 \times 3$.
2. **Form $\Sigma^+$:** If $\sigma_1, \sigma_2 > 0$, then:

$$
\Sigma^+ = \begin{bmatrix} 1/\sigma_1 & 0 \\ 0 & 1/\sigma_2 \\ 0 & 0 \end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\Sigma^+$ | Pseudo-inverse of the singular value matrix ($3 \times 2$) | Reciprocals of non-zero singular values placed on the diagonal; transposed shape |
| $1/\sigma_1$ | Reciprocal of the first singular value | Inverts the scaling of the dominant direction |
| $1/\sigma_2$ | Reciprocal of the second singular value | Inverts the scaling of the second direction |

3. **Result:** 

$$
A^+ = V \Sigma^+ U^T \approx \begin{bmatrix} -0.0056 & 0.0722 \\ 0.0222 & 0.0444 \\ 0.0556 & -0.0556 \end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $A^+$ | Moore-Penrose pseudo-inverse of $A$ ($3 \times 2$) | Generalizes the matrix inverse to non-square matrices |
| $V$ | Right singular vectors from SVD | Orthogonal basis for the row space of $A$ |
| $\Sigma^+$ | Pseudo-inverse of $\Sigma$ | Contains reciprocals of non-zero singular values |
| $U^T$ | Transpose of left singular vectors | Orthogonal basis for the column space of $A$ |

```python
import numpy as np

X = np.array([[4, 11, 14], 
              [8, 7, -2]])

# In Python, predicting weights w in OLS Regression secretly relies on SVD!
# The pinv() function builds V * Sigma^+ * U^T
pseudo_inverse = np.linalg.pinv(X)
```

---

## 4. Matrix Norms

Matrix norms boil the absolute "magnitude" of a massive web of numbers down into a single scalar metric. 

### Spectral Norm (2-Norm)
The largest singular value: $\sigma_1$. It tells you the absolute maximum severity by which the matrix stretches space in its most dominant direction.

### Frobenius Norm
The absolute bedrock norm in Deep Learning optimization. It is essentially the standard Euclidean ($L_2$) distance unrolled across an entire matrix.

$$
\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2} = \sqrt{\text{Tr}(A^T A)} = \sqrt{\sum \sigma_i^2}
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $\|A\|_F$ | The Frobenius norm of matrix $A$ | Measures the overall "size" of the matrix using Euclidean distance |
| $a_{ij}$ | Individual matrix entries | Each element contributes quadratically to the total norm |
| $\text{Tr}(A^T A)$ | Trace of $A^T A$ | Equals the sum of squared singular values via SVD |
| $\sigma_i$ | Singular values of $A$ | Alternative computation: norm is the root sum of squared singular values |

**Deep Learning Connection:** When you use "Weight Decay" in PyTorch (L2 Regularization on the model's weights), the optimizer is actively adding a massive penalty proportional to the **Frobenius Norm** of the weight matrices to force the network to remain simple.

---

## 5. Comprehensive Worked Example

**Matrix:** 

$$
A = \begin{bmatrix} 0 & 1 & 1 \\ \sqrt{2} & 2 & 0 \\ 0 & 1 & 1 \end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $A$ | Comprehensive example matrix ($3 \times 3$) | Rank-2 matrix used to demonstrate SVD metrics and matrix norms |
| $0, 1, 1, \sqrt{2}, 2, 0, \dots$ | Entries of $A$ | Contains repeated rows leading to rank deficiency |

**1. SVD Metrics:**
Running SVD yields $\sigma_1 = 2\sqrt{2}, \sigma_2 = \sqrt{2}, \sigma_3 = 0$.
The exact Rank is $2$ (since we have exactly two non-zero singular values).

**2. Norm Calculations In Action:**

* **Spectral Norm ($\|A\|_2$):** Simply the dominant singular value: $\sigma_1 = 2\sqrt{2} \approx 2.828$.
* **Frobenius Norm ($\|A\|_F$):** $\sqrt{\sigma_1^2 + \sigma_2^2 + \sigma_3^2} = \sqrt{8 + 2 + 0} = \sqrt{10}$.

---

## 6. Modern SVD in Deep Learning

* **Weight Initialization:** Research shows initializing deep network weight matrices as purely orthogonal matrices $U$ and $V^T$ generated via SVD guarantees perfect, gradient-preserving information flow entirely free from vanishing gradients early in training.
* **Regularization:** In Ridge Regression, adding an $L_2$ penalty mathematically shifts all singular values away from zero, violently neutralizing singularities and allowing the Pseudo-Inverse $A^+$ to compute stably.

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 12: SVD](Lecture%2012%20Singular%20Value%20Decomposition.md) — The core factorization $A = U\Sigma V^T$ that these properties and applications build upon
- **Next:** [Lecture 14: Low Rank](Lecture%2014%20Low%20Rank%20Approximations.md) — Truncates small singular values for efficient compression and approximation
- **Related:** [Lecture 27: Gram-Schmidt](Lecture%2027%20Gram%20Schmidt%20Process.md) — Constructs orthonormal bases; the $U$ and $V$ matrices in SVD are orthonormal
