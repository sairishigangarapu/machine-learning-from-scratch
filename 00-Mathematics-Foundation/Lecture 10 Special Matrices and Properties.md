# Positive Definite Matrices

This lecture builds upon previous concepts of eigenvalues and eigenvectors to discuss **positive definite matrices**. These are crucial in convex optimization and machine learning for analyzing objective functions.

---

## 1. Quadratic Form

Let $A$ be an $n \times n$ real symmetric matrix and $x$ be an $n$-dimensional vector.

The expression $Q(x)$ is called the quadratic form associated with matrix $A$:

$$
Q(x) = x^T A x
$$

### Expansion Example

For a $3 \times 3$ symmetric matrix $A$ and vector $x = [x_1, x_2, x_3]^T$:

$$
A = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix}
$$

Since $A$ is symmetric, $a_{ij} = a_{ji}$. The quadratic form expands to:

$$
Q(x) = \sum_{i=1}^n a_{ii}x_i^2 + \sum_{i \neq j} a_{ij}x_ix_j
$$

$$
Q(x) = a_{11}x_1^2 + a_{22}x_2^2 + a_{33}x_3^2 + 2a_{12}x_1x_2 + 2a_{13}x_1x_3 + 2a_{23}x_2x_3
$$

**Example:**

Given quadratic form $Q(x) = x_1^2 - 2x_2^2 + 4x_1x_3$.

The corresponding matrix $A$ is:

$$
A = \begin{bmatrix} 1 & 0 & 2 \\ 0 & -2 & 0 \\ 2 & 0 & 0 \end{bmatrix}
$$

- Diagonal elements come from squared terms ($x_i^2$).
- Off-diagonal elements come from cross terms ($x_ix_j$), divided by 2.

---

## 2. Rayleigh Quotient

For a real symmetric matrix $A$, the Rayleigh Quotient is defined as:

$$
R(A, x) = \frac{x^T A x}{x^T x}
$$

where $x \neq 0$.

### Properties

1. **Scale Invariance:** $R(A, \alpha x) = R(A, x)$ for any scalar $\alpha \neq 0$.
2. **Eigenvalue Relation:** If $x$ is an eigenvector with eigenvalue $\lambda$, then $R(A, x) = \lambda$.
3. **Bounds:** For any unit vector $x$ ($\|x\|=1$):
   
   $$
   \lambda_{\min}(A) \le x^T A x \le \lambda_{\max}(A)
   $$

   - The minimum value is the smallest eigenvalue $\lambda_{\min}$.
   - The maximum value is the largest eigenvalue $\lambda_{\max}$.
   - Equality holds if and only if $x$ is the corresponding eigenvector.

---

## 3. Positive Definite & Semi-Definite Matrices

A symmetric matrix $A$ is defined based on the sign of its quadratic form $x^T A x$ for all non-zero vectors $x \in \mathbb{R}^n$.

| **Type** | **Condition (for all $x \neq 0$)** | **Eigenvalue Condition** | **Pivot Test** |
| :--- | :--- | :--- | :--- |
| **Positive Definite (PD)** | $x^T A x > 0$ | All $\lambda_i > 0$ | All pivots > 0 |
| **Positive Semi-Definite (PSD)** | $x^T A x \ge 0$ | All $\lambda_i \ge 0$ | All pivots $\ge 0$ |

**Example of PSD Matrix:**

$$
A = \begin{bmatrix} 1 & 3 \\ 3 & 10 \end{bmatrix}
$$

$$
Q(x) = x_1^2 + 10x_2^2 + 6x_1x_2 = (x_1 + 3x_2)^2 + x_2^2
$$

Since $Q(x)$ is a sum of squares, $Q(x) \ge 0$ for all real $x$. Thus, $A$ is PSD. Since it can only be 0 if $x=0$, it is also PD.

---

## 4. Tests for Positive Definiteness

### Eigenvalue Test

A symmetric matrix is PD (or PSD) iff all its eigenvalues are positive (or non-negative).

### Pivot / Principal Minor Test (Sylvester's Criterion)

Check the determinants of the upper-left sub-matrices (principal minors).


### Parameter Example

For $A = \begin{bmatrix} 2 & -1 & b \\ -1 & 2 & -1 \\ b & -1 & 2 \end{bmatrix}$ to be PSD:

The condition leads to $b \in [-1, 2]$.

- Strict inequality ($b \in (-1, 2)$) makes it Positive Definite.

---

## 5. Important Properties for Machine Learning

1. **Gram Matrix Property:**
   For any real matrix $A$ (not necessarily symmetric), the matrix $A^T A$ is always **Positive Semi-Definite**.
   - If the null space of $A$ contains only the zero vector (i.e., $A$ has full rank), then $A^T A$ is **Positive Definite**.
   - *Application:* This is often used to convert a linear system $Ax=b$ into the normal equations $A^T Ax = A^T b$, ensuring the coefficient matrix is symmetric and PSD.

2. **Regularization:**
   If $A$ is PSD and $\epsilon > 0$, then $A + \epsilon I$ is **Positive Definite**.
   - Adding a small value to the diagonal shifts all eigenvalues up by $\epsilon$, making them strictly positive.

3. **Geometry:**
   The iso-contours of the quadratic form $x^T A x = c$ for a PD matrix define an ellipsoid.
   - **Axes directions:** Given by the eigenvectors of $A$.
   - **Axes lengths:** Proportional to $1/\sqrt{\lambda_i}$.
