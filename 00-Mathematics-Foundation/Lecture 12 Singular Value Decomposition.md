## Singular Value Decomposition (SVD)

*The Universal Factorization for All Matrices*

---

## 1. Introduction: Why SVD?

While **Spectral Decomposition** is a powerful tool, it is strictly limited to square, diagonalizable matrices (and is most elegant for symmetric ones). However, real-world data is rarely square.

**Singular Value Decomposition (SVD)** is the generalization of eigendecomposition to **any** $m \times n$ matrix. Whether the matrix is square, rectangular, tall, or fat, SVD provides a consistent, orthogonal coordinate system that reveals the matrix's underlying geometry.

---

## 2. Formal Definition

For any real $m \times n$ matrix $A$, there exists a factorization:


$$\boxed{A = U \Sigma V^T}$$


Where:

* **$U$ ($m \times m$):** An orthogonal matrix whose columns are the **left singular vectors**. These are the orthonormal eigenvectors of $A A^T$.
* **$V$ ($n \times n$):** An orthogonal matrix whose columns are the **right singular vectors**. These are the orthonormal eigenvectors of $A^T A$.
* **$\Sigma$ ($m \times n$):** A rectangular diagonal matrix containing the **singular values** $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$ (where $r$ is the rank of $A$).

### Singular Values vs. Eigenvalues

The singular values $\sigma_i$ are the positive square roots of the non-zero eigenvalues of $A^T A$ (or $A A^T$):


$$\sigma_i = \sqrt{\lambda_i}$$

---

## 3. Mathematical Foundations: $A^T A$ and $A A^T$

To find the components of SVD, we analyze two related symmetric matrices:

1. **Right Singular Vectors ($V$):** Found by eigendecomposing the $n \times n$ matrix $A^T A$. Since $A^T A$ is symmetric and positive semi-definite, its eigenvalues are $\ge 0$.
2. **Left Singular Vectors ($U$):** Found by eigendecomposing the $m \times m$ matrix $A A^T$.
3. **Relation:** $A \mathbf{v}_i = \sigma_i \mathbf{u}_i$. This means once you have the singular values and $V$, you can often derive $U$ without a second full eigendecomposition.

---

## 4. Geometric Interpretation

SVD decomposes any linear transformation into three distinct geometric steps:

1. **Rotation ($V^T$):** Rotating the input vector into the principal axes of the transformation.
2. **Scaling ($\Sigma$):** Stretching or shrinking along these axes based on the singular values.
3. **Rotation ($U$):** A final rotation in the output space.

---

## 5. Comprehensive Worked Example (Rectangular Matrix)

**Matrix:** $A = \begin{bmatrix} 4 & 11 & 14 \\ 8 & 7 & -2 \end{bmatrix}$ ($2 \times 3$ matrix)

**Step 1: Find $A^T A$ and Singular Values**


$$A^T A = \begin{bmatrix} 80 & 100 & 40 \\ 100 & 170 & 140 \\ 40 & 140 & 200 \end{bmatrix}$$


The eigenvalues of $A^T A$ are $\lambda_1 = 360, \lambda_2 = 90, \lambda_3 = 0$.

* **Singular Values:** $\sigma_1 = \sqrt{360} = 6\sqrt{10}$, $\sigma_2 = \sqrt{90} = 3\sqrt{10}$.

**Step 2: Find Right Singular Vectors ($V$)**
Find the orthonormal eigenvectors for $A^T A$:

* $\mathbf{v}_1 = \frac{1}{3} [1, 2, 2]^T$
* $\mathbf{v}_2 = \frac{1}{3} [-2, -1, 2]^T$
* $\mathbf{v}_3 = \frac{1}{3} [2, -2, 1]^T$

$$V = \begin{bmatrix} 1/3 & -2/3 & 2/3 \\ 2/3 & -1/3 & -2/3 \\ 2/3 & 2/3 & 1/3 \end{bmatrix}$$



**Step 3: Find Left Singular Vectors ($U$)**
Using the relation $\mathbf{u}_i = \frac{1}{\sigma_i} A \mathbf{v}_i$:

* $\mathbf{u}_1 = \frac{1}{6\sqrt{10}} A \mathbf{v}_1 = \frac{1}{\sqrt{10}} [3, 1]^T$
* $\mathbf{u}_2 = \frac{1}{3\sqrt{10}} A \mathbf{v}_2 = \frac{1}{\sqrt{10}} [1, -3]^T$

$$U = \begin{bmatrix} 3/\sqrt{10} & 1/\sqrt{10} \\ 1/\sqrt{10} & -3/\sqrt{10} \end{bmatrix}$$



**Step 4: Construct $\Sigma$**
$\Sigma$ matches the dimensions of $A$ ($2 \times 3$):


$$\Sigma = \begin{bmatrix} 6\sqrt{10} & 0 & 0 \\ 0 & 3\sqrt{10} & 0 \end{bmatrix}$$

---

## 6. Important Properties of SVD

* **Rank Determination:** The number of non-zero singular values equals the rank of the matrix $A$.
* **Matrix Norms:** The largest singular value $\sigma_1$ is the spectral norm of $A$.
* **Condition Number:** $\kappa = \sigma_{\max} / \sigma_{\min}$ measures the numerical stability of the matrix.
* **Low-Rank Approximation:** The best rank-$k$ approximation of $A$ (in terms of Frobenius norm) is found by keeping only the top $k$ singular values (Eckart-Young-Mirsky Theorem).

$$A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$



---

## 7. Machine Learning Applications

* **Data Compression:** In image processing, we can reconstruct an image using only the top $k$ singular vectors, significantly reducing storage.
* **Latent Semantic Analysis (LSA):** Used in NLP to find relationships between documents and terms by reducing the rank of a term-document matrix.
* **Pseudo-Inverse:** SVD is used to calculate the Moore-Penrose inverse ($A^+$) for solving non-square linear systems: $A^+ = V \Sigma^+ U^T$.
* **Collaborative Filtering:** SVD is the engine for movie recommendation systems (like the Netflix Prize), identifying latent features of users and items.

---
