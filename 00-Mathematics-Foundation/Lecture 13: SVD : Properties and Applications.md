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

SVD provides an orthonormal basis for the four fundamental subspaces of an $m \times n$ matrix $A$ (where $\text{rank}(A) = r$):

| Subspace | Basis | Dimension |
| --- | --- | --- |
| **Range Space** $\text{Range}(A)$ | First $r$ columns of $U$ | $r$ |
| **Null Space** $\text{Null}(A)$ | Last $n-r$ columns of $V$ | $n-r$ |
| **Row Space** $\text{Range}(A^T)$ | First $r$ columns of $V$ | $r$ |
| **Left Null Space** $\text{Null}(A^T)$ | Last $m-r$ columns of $U$ | $m-r$ |

> **Practical Rule:** The columns of $V$ associated with zero (or near-zero) singular values form the basis for the **Null Space**, representing directions that the transformation "collapses."

---

## 3. The Moore-Penrose Pseudo-Inverse ($A^+$)

For non-square or singular matrices where a standard inverse doesn't exist, we use the **Pseudo-Inverse**:


$$\boxed{A^+ = V \Sigma^+ U^T}$$


Where $\Sigma^+$ is formed by transposing $\Sigma$ and replacing every non-zero singular value $\sigma_i$ with its reciprocal $1/\sigma_i$. If $\sigma_i = 0$, it remains $0$.

### Worked Example: Pseudo-Inverse

**Given:** $A = \begin{bmatrix} 4 & 11 & 14 \\ 8 & 7 & -2 \end{bmatrix}$ (Rank 2)

1. **Identify Components:** $U$ is $2 \times 2$, $\Sigma$ is $2 \times 3$, and $V$ is $3 \times 3$.
2. **Form $\Sigma^+$:** If $\sigma_1, \sigma_2 > 0$, then $\Sigma^+ = \begin{bmatrix} 1/\sigma_1 & 0 \\ 0 & 1/\sigma_2 \\ 0 & 0 \end{bmatrix}$.
3. **Result:** $A^+ = V \Sigma^+ U^T \approx \begin{bmatrix} -0.0056 & 0.0722 \\ 0.0222 & 0.0444 \\ 0.0556 & -0.0556 \end{bmatrix}$.

---

## 4. Matrix Norms

Matrix norms quantify the "size" of a matrix or the strength of its transformation.

### Induced Operator Norms ($p$-norms)

* **1-Norm (Max Column Sum):** Maximum absolute sum of entries in any single column.
* **$\infty$-Norm (Max Row Sum):** Maximum absolute sum of entries in any single row.
* **2-Norm (Spectral Norm):** The largest singular value, $\sigma_1$. It represents the maximum possible stretch factor.

### Frobenius Norm

The most common norm in ML optimization, representing the "Euclidean distance" for matrices:


$$\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2} = \sqrt{\text{Tr}(A^T A)} = \sqrt{\sum \sigma_i^2}$$

---

## 5. Comprehensive Worked Example

**Matrix:** $A = \begin{bmatrix} 0 & 1 & 1 \\ \sqrt{2} & 2 & 0 \\ 0 & 1 & 1 \end{bmatrix}$

**1. SVD Metrics:**

* **Singular Values:** $\sigma_1 = 2\sqrt{2}, \sigma_2 = \sqrt{2}, \sigma_3 = 0$.
* **Rank:** $2$.

**2. Norm Calculations:**

* **1-Norm:** Column sums are $\{\sqrt{2}, 4, 2\}$. Max = $4$.
* **$\infty$-Norm:** Row sums are $\{2, 2+\sqrt{2}, 2\}$. Max = $2+\sqrt{2}$.
* **Spectral Norm ($\|A\|_2$):** $\sigma_1 = 2\sqrt{2} \approx 2.828$.
* **Frobenius Norm ($\|A\|_F$):** $\sqrt{\sigma_1^2 + \sigma_2^2 + \sigma_3^2} = \sqrt{8 + 2 + 0} = \sqrt{10}$.

---

## 6. SVD in Machine Learning

* **Range Space Discovery:** The first $r$ columns of $U$ give the directions of the data points in the target space.
* **Regularization:** In Ridge Regression, adding $\lambda I$ prevents the singular values from being too small, stabilizing the calculation of $A^+$.
* **Unitary Invariance:** Many matrix norms (like Frobenius and Spectral) are "unitary invariant," meaning $\|UAV\| = \|A\|$. This is critical because it shows that rotating the data doesn't change its fundamental "size" or information content.

---
