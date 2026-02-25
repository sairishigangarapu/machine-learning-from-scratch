## Positive Definite Matrices

*Essential Mathematics for ML — Structured Notes*

---

## 1. Quadratic Form

In machine learning, we often use quadratic forms to represent loss functions or the variance of data. Let $A$ be an $n \times n$ real symmetric matrix and $\mathbf{x}$ be an $n$-dimensional vector.

The expression $Q(\mathbf{x})$ is defined as the quadratic form associated with matrix $A$:


$$Q(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$$

### Expansion and Matrix Mapping

For a symmetric matrix, diagonal elements represent the coefficients of squared terms ($x_i^2$), and off-diagonal elements represent half the coefficients of cross-terms ($x_i x_j$).

**Example:**
Given $Q(\mathbf{x}) = x_1^2 - 2x_2^2 + 4x_1x_3$.
The corresponding symmetric matrix $A$ is:


$$A = \begin{bmatrix} 1 & 0 & 2 \\ 0 & -2 & 0 \\ 2 & 0 & 0 \end{bmatrix}$$


*Note: The coefficient $4$ for $x_1x_3$ is split into $a_{13}=2$ and $a_{31}=2$ to maintain symmetry.*

---

## 2. Rayleigh Quotient

The Rayleigh Quotient provides a way to relate a matrix to its eigenvalues through a scalar ratio. For a real symmetric matrix $A$ and $\mathbf{x} \neq 0$:


$$R(A, \mathbf{x}) = \frac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T \mathbf{x}}$$

### Key Properties

1. **Scale Invariance:** $R(A, \alpha \mathbf{x}) = R(A, \mathbf{x})$ for any scalar $\alpha \neq 0$.
2. **Eigenvalue Relation:** If $\mathbf{x}$ is an eigenvector with eigenvalue $\lambda$, then $R(A, \mathbf{x}) = \lambda$.
3. **Bounds:** For any unit vector $\mathbf{x}$ ($\|\mathbf{x}\|=1$):

$$\lambda_{\min}(A) \le \mathbf{x}^T A \mathbf{x} \le \lambda_{\max}(A)$$



The quadratic form is bounded by the smallest and largest eigenvalues of $A$.

---

## 3. Defining Positive Definite (PD) and Semi-Definite (PSD)

A symmetric matrix $A$ is categorized based on the sign of its quadratic form for all non-zero vectors $\mathbf{x} \in \mathbb{R}^n$.

| **Type** | **Condition (for all $\mathbf{x} \neq 0$)** | **Eigenvalue Condition** | **Pivot Test** |
| --- | --- | --- | --- |
| **Positive Definite (PD)** | $\mathbf{x}^T A \mathbf{x} > 0$ | All $\lambda_i > 0$ | All pivots $> 0$ |
| **Positive Semi-Definite (PSD)** | $\mathbf{x}^T A \mathbf{x} \ge 0$ | All $\lambda_i \ge 0$ | All pivots $\ge 0$ |

**Example of Verification:**
For $A = \begin{bmatrix} 1 & 3 \\ 3 & 10 \end{bmatrix}$, the quadratic form is:
$Q(\mathbf{x}) = x_1^2 + 10x_2^2 + 6x_1x_2 = (x_1 + 3x_2)^2 + x_2^2$.
Since $Q(\mathbf{x})$ is a sum of squares, $Q(\mathbf{x}) \ge 0$. It is 0 only if $\mathbf{x}=0$, making $A$ **Positive Definite**.

---

## 4. Tests for Positive Definiteness

### Eigenvalue Test

A symmetric matrix is PD (or PSD) if and only if all its eigenvalues are strictly positive (or non-negative).

### Pivot / Principal Minor Test (Sylvester's Criterion)

Check the determinants of all upper-left sub-matrices (leading principal minors).
For $A = \begin{bmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{bmatrix}$:

1. **$1 \times 1$:** $\det([2]) = 2 > 0$
2. **$2 \times 2$:** $\det \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix} = (4 - 1) = 3 > 0$
3. **$3 \times 3$:** $\det(A) = 4 > 0$
Since all principal minors are positive, **$A$ is Positive Definite**.

### Parameter Dependency Example

For $A = \begin{bmatrix} 2 & -1 & b \\ -1 & 2 & -1 \\ b & -1 & 2 \end{bmatrix}$ to be **PSD**, calculations show $b$ must fall in the range $b \in [-1, 2]$.

* If $b \in (-1, 2)$, the matrix is strictly **Positive Definite**.

---

## 5. Important Properties for Machine Learning

### 1. Gram Matrix Property

For any real matrix $A$ (not necessarily symmetric or square), the product $A^T A$ is **always Positive Semi-Definite**.

* **Full Rank Case:** If $A$ has full column rank, $A^T A$ is strictly **Positive Definite**.
* **Application:** In Linear Regression, we solve the normal equations $(A^T A)\mathbf{x} = A^T \mathbf{b}$. The PD nature of $A^T A$ ensures a unique solution exists.

### 2. Regularization (Tikhonov Regularization)

If $A$ is PSD and $\epsilon > 0$, then $A + \epsilon I$ is strictly **Positive Definite**.

* **Mechanism:** Adding $\epsilon I$ shifts all eigenvalues up by $\epsilon$ ($\lambda_{new} = \lambda_{old} + \epsilon$).
* **Application:** This is used in Ridge Regression to ensure the matrix inverse is stable and well-conditioned.

### 3. Geometric Interpretation

The iso-contours of the quadratic form $\mathbf{x}^T A \mathbf{x} = c$ for a PD matrix define an **ellipsoid** in $n$-dimensional space.

* **Axes Directions:** Defined by the **eigenvectors** of $A$.
* **Axes Lengths:** Inversely proportional to the square root of eigenvalues ($1/\sqrt{\lambda_i}$).
* **Convexity:** If $A$ is PD, the function $f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ is strictly convex, guaranteeing a single global minimum.

---

## 6. Implementation in Python (NumPy)

```python
import numpy as np

# Define a symmetric matrix
A = np.array([[2, -1, 0], 
              [-1, 2, -1], 
              [0, -1, 2]])

# 1. Eigenvalue Test
eigenvalues = np.linalg.eigvals(A)
is_pd = np.all(eigenvalues > 0)

# 2. Cholesky Decomposition (Alternative Test)
# Cholesky fails if the matrix is not Positive Definite
try:
    L = np.linalg.cholesky(A)
    print("Matrix is Positive Definite")
except np.linalg.LinAlgError:
    print("Matrix is not Positive Definite")

```

---
