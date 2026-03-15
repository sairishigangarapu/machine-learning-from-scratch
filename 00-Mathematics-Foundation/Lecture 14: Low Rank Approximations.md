## Low-Rank Approximation

*Essential Mathematics for ML — Advanced Structured Notes*

---

## 1. Matrix Rank and Efficiency

The **rank** of an $m \times n$ matrix $A$ is the dimension of its range space, representing the number of linearly independent rows or columns. In the context of big data, full-rank matrices are often computationally expensive to store and process.

### The Factorization Perspective

If $\text{rank}(A) = r$, the matrix can be factorized into two smaller matrices:


$$\boxed{A = BC^T}$$


Where:

* **$B$ ($m \times r$):** A basis for the range space.
* **$C^T$ ($r \times n$):** The coordinates in that basis.

> **Storage Motivation:** Consider a $50 \times 100$ matrix with rank $r = 20$.
> * Original storage: $50 \times 100 = 5,000$ elements.
> * Factorized storage: $(50 \times 20) + (100 \times 20) = 3,000$ elements.
> * **Result:** 40% memory saving by leveraging low-rank structure.
> 
> 

---

## 2. Defining the Low-Rank Approximation Problem

The goal is to find a matrix $A_k$ of rank $k$ (where $k < \text{rank}(A)$) that is the "closest" possible approximation to $A$ under a specific norm (usually Frobenius).

### The Optimization Objective

$$\min_{A_k} \|A - A_k\|_F \quad \text{subject to} \quad \text{rank}(A_k) \le k$$

**Note on Non-Convexity:** This problem is mathematically **non-convex**. For instance, the average of two different rank-1 matrices is often a rank-2 matrix. This makes the search for a global minimum difficult using standard gradient methods.

---

## 3. The SVD Solution (Eckart-Young-Mirsky Theorem)

Despite the non-convexity, **Singular Value Decomposition** provides a closed-form, globally optimal solution. If $A = U \Sigma V^T$, the best rank-$k$ approximation is found by truncating the SVD:


$$\boxed{A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T}$$


In matrix form:


$$A_k = U \Sigma_k V^T$$


where $\Sigma_k$ contains only the top $k$ singular values, and the rest are set to zero.

### Measuring Quality

The "energy" or information retained by the approximation is calculated as:


$$\text{Quality Score} = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2}$$

---

## 4. Comprehensive Worked Example

**Original Matrix ($3 \times 3$):**


$$A = \begin{bmatrix} 3 & 2 & 1 \\ 0 & 2 & 1 \\ 0 & 0 & 1 \end{bmatrix}$$

**1. SVD Extraction:**
$\sigma_1 \approx 4.12, \sigma_2 \approx 1.73, \sigma_3 \approx 0.70$.

**2. Rank-2 Approximation ($A_2$):**
By zeroing out $\sigma_3$ and re-multiplying $U \Sigma_2 V^T$, we get:


$$A_2 \approx \begin{bmatrix} 2.99 & 2.01 & 0.98 \\ 0.01 & 1.99 & 1.02 \\ 0.00 & 0.01 & 0.99 \end{bmatrix}$$


The error $\|A - A_2\|$ is minimized, and we have successfully captured the most significant variance with one less dimension.

---

## 5. Application: Image Compression

Digital images are stored as matrices where entries represent pixel intensity.

* **Input Image:** A high-resolution photo might have rank $266$.
* **Rank 1-10:** We see only the "ghost" or dominant structural patterns of the image.
* **Rank 25-50:** The image becomes clearly recognizable.
By saving only the first 50 singular values and vectors, we can achieve high-fidelity reconstruction while saving over 80% of disk space.

---

## 6. Implementation in Python (NumPy)

```python
import numpy as np

# Original Matrix
A = np.array([[3, 2, 1], [0, 2, 1], [0, 0, 1]])

# Full SVD
U, S, Vt = np.linalg.svd(A)

# Rank-2 Approximation
k = 2
Sk = np.diag(S[:k])
Ak = U[:, :k] @ Sk @ Vt[:k, :]

print("Rank-2 Approximation Matrix:\n", Ak)

```

---
