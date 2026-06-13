## Singular Value Decomposition (SVD)

*The Universal Factorization for All Matrices*

---

## 1. Introduction: Why SVD?

### Motivation and Intuition
**Spectral Decomposition** is a gorgeous mathematical tool, but it has a fatal, crippling flaw: it strictly requires the matrix to be a perfect square ($n \times n$). 

In Machine Learning, your data matrix $X$ is practically never square. You might have 10,000 patients (rows) and 50 blood test features (columns). Eigendecomposition is mathematically impossible on a $10000 \times 50$ rectangle.

**Singular Value Decomposition (SVD)** is the Swiss Army Knife of Linear Algebra. It is the generalization of eigendecomposition that works on **every single matrix in existence**. Whether the matrix is square, a tall rectangle, a fat rectangle, singular, or defective—SVD provides a flawless, orthogonal coordinate system that tears the matrix apart to reveal its hidden geometric structure.

---

## 2. Formal Definition

For literally any real $m \times n$ matrix $A$, there exists an exact factorization:

$$
\boxed{A = U \Sigma V^T}
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $A$ | Any real $m \times n$ matrix | SVD works for all matrices regardless of shape or rank |
| $U$ | $m \times m$ orthogonal matrix of left singular vectors | Defines the output coordinate system after scaling |
| $\Sigma$ | $m \times n$ rectangular diagonal matrix of singular values | Contains the scaling factors $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$ |
| $V^T$ | $n \times n$ orthogonal matrix of right singular vectors | Defines the input coordinate system before scaling |

* **$V^T$ ($n \times n$):** An orthogonal matrix of **Right Singular Vectors**.
* **$\Sigma$ ($m \times n$):** A rectangular diagonal matrix containing **Singular Values** $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$.
* **$U$ ($m \times m$):** An orthogonal matrix of **Left Singular Vectors**.

### Geometric Interpretation
SVD proves that any bizarre matrix transformation in the universe can be broken down into three perfectly clean, sequential geometric steps:

1. **$V^T$ (Rotation):** Spin the features into a new coordinate frame.
2. **$\Sigma$ (Scaling):** Stretch or squash those aligned axes.
3. **$U$ (Rotation):** Spin the output into the final embedding space.

---

## 3. Mathematical Foundations: The Bridge to Eigenvalues

How do we find SVD? By cleverly turning our rectangular matrix $A$ into two temporary symmetric square matrices: $A^T A$ and $A A^T$.

1. **Right Singular Vectors ($V$):** Found precisely by running standard eigendecomposition on $A^T A$.
2. **Left Singular Vectors ($U$):** Found by eigendecomposing $A A^T$.
3. **Singular Values ($\sigma_i$):** Just the positive square roots of those eigenvalues:

$$
\sigma_i = \sqrt{\lambda_i}
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $\sigma_i$ | The $i$-th singular value of matrix $A$ | Measures the "strength" of the $i$-th principal component |
| $\sqrt{\cdot}$ | The positive square root function | Ensures singular values are non-negative |
| $\lambda_i$ | Eigenvalue of $A^T A$ or $A A^T$ | Relates rectangular matrix properties to symmetric matrix eigenvalues |

---

## 4. Comprehensive Worked Example (Rectangular Matrix)

**Matrix:** $A = \begin{bmatrix} 4 & 11 & 14 \\ 8 & 7 & -2 \end{bmatrix}$ (A $2 \times 3$ matrix)

**Step 1: Singular Values via $A^T A$**

$$
A^T A = \begin{bmatrix} 80 & 100 & 40 \\ 100 & 170 & 140 \\ 40 & 140 & 200 \end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $A^T A$ | Product of $A^T$ and $A$ | $3 \times 3$ symmetric matrix whose eigenvectors give the right singular vectors $V$ |
| $80, 100, 40, \dots$ | Entries of $A^T A$ | Inner products of the columns of $A$; encode feature correlations |

The eigenvalues of this symmetric matrix are $\lambda = \{360, 90, 0\}$.

* **Singular Values:** $\sigma_1 = \sqrt{360} \approx 18.97$, $\sigma_2 = \sqrt{90} \approx 9.48$.

**Step 2: Right Singular Vectors ($V$)**
Find the eigenvectors for $A^T A$:

$$
V = \begin{bmatrix} 1/3 & -2/3 & 2/3 \\ 2/3 & -1/3 & -2/3 \\ 2/3 & 2/3 & 1/3 \end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $V$ | Right singular vector matrix ($3 \times 3$) | Orthogonal matrix whose columns are eigenvectors of $A^T A$ |
| $1/3, 2/3, \dots$ | Components of the right singular vectors | Define the orthonormal basis for the input (row) space of $A$ |

**Step 3: Construct The Diagonal ($\Sigma$)**
$\Sigma$ strictly matches the shape of $A$ ($2 \times 3$):

$$
\Sigma = \begin{bmatrix} 18.97 & 0 & 0 \\ 0 & 9.48 & 0 \end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\Sigma$ | Rectangular diagonal singular value matrix ($2 \times 3$) | Contains singular values $\sigma_1, \sigma_2$ on the diagonal |
| $18.97$ | First singular value $\sigma_1$ | Largest scaling factor; square root of the largest eigenvalue of $A^T A$ |
| $9.48$ | Second singular value $\sigma_2$ | Second scaling factor; square root of the second eigenvalue of $A^T A$ |

```python
import numpy as np

A = np.array([[4, 11,  14], 
              [8,  7, -2]])

# NumPy natively computes the full SVD in one line
U, S, Vt = np.linalg.svd(A)

print("Singular Values (S):", S) # Approx [18.97, 9.48]
```

---

## 5. Important Properties governing SVD

* **Rank Determination:** By simply counting the number of strictly non-zero singular values in $\Sigma$, we have a mathematically infallible way to discover the true, exact Rank of matrix $A$.
* **Spectral Matrix Norm ($\|A\|_2$):** Want to know the absolute maximum "stretch" this matrix applies to any input? Look at the top left value of $\Sigma$. The largest singular value $\sigma_1$ is the Spectral Norm. 
* **Condition Number:** $\kappa = \sigma_{\max} / \sigma_{\min}$. If $\sigma_{\min}$ is terrifyingly close to zero, $\kappa \to \infty$. This implies the matrix is "ill-conditioned" and computers will suffer massive floating-point rounding errors attempting to invert it.

---

## 6. The SVD Power-User Case: ML Applications

SVD is the most important decomposition in modern data science:

* **[Principal Component Analysis (PCA):](Lecture%2016%20Principal%20Component%20Analysis.md)** Most high-performance PCA implementations (like `sklearn`) use SVD internally because it is numerically more stable than eigendecomposition.
* **[The Moore-Penrose Pseudo-Inverse:](Lecture%2021%20Least%20Square%20Approximation%20and%20Minimum%20Norm%20Solution.md)** As seen in Lecture 21, we solve impossible linear systems using $A^{+} = V \Sigma^{-1} U^T$.
* **Low-Rank Approximation:** As explored in Lecture 14, SVD allows us to compress massive datasets (like images or user-item matrices in Recommenders) by throwing away small singular values.

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 11: Spectral](Lecture%2011%20Spectral%20decomposition.md) — Eigendecomposition of symmetric matrices is the mathematical bridge to SVD
- **Next:** [Lecture 13: SVD Properties](Lecture%2013%20SVD%20%3A%20Properties%20and%20Applications.md) — Explores matrix norms, pseudo-inverse, and subspace structure via SVD
- **Related:** [Lecture 16: PCA](Lecture%2016%20Principal%20Component%20Analysis.md) — SVD is the computational backbone of modern PCA implementations
- **Related:** [Lecture 21: Least Squares](Lecture%2021%20Least%20Square%20Approximation%20and%20Minimum%20Norm%20Solution.md) — The pseudo-inverse from SVD solves overdetermined and underdetermined systems
- **Related:** [Lecture 14: Low Rank](Lecture%2014%20Low%20Rank%20Approximations.md) — Truncating small singular values yields optimal low-rank approximations
