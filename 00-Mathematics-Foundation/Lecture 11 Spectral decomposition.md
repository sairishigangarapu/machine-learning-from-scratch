## Spectral Decomposition (Eigendecomposition)

*Essential Mathematics for ML — Advanced Structured Notes*

---

## 1. Theoretical Foundation

Spectral decomposition, also known as **Eigendecomposition**, is the process of factorizing a square matrix into a form that reveals its "spectrum" (eigenvalues). In Machine Learning, this isn't just a matrix trick; it is the fundamental mechanism for **representation learning**, allowing us to change the basis of our data to a coordinate system where features are uncorrelated.

### Formal Definition

Let $A \in \mathbb{R}^{n \times n}$ be a square matrix. If $A$ has $n$ linearly independent eigenvectors $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n\}$, then $A$ can be decomposed into:


$$\boxed{A = PDP^{-1}}$$


Where:

* **$P$ (Modal Matrix):** An $n \times n$ matrix where the $i$-th column is the eigenvector $\mathbf{v}_i$.
* **$D$ (Spectral Matrix):** A diagonal matrix where $D_{ii} = \lambda_i$, the eigenvalue associated with $\mathbf{v}_i$.
* **$P^{-1}$:** The change-of-basis matrix that maps the standard basis to the eigenvector basis.

---

## 2. Necessary and Sufficient Conditions

Not every square matrix can be decomposed in this way. A matrix $A$ is **diagonalizable** if and only if:

1. **Algebraic Multiplicity = Geometric Multiplicity:** For every eigenvalue $\lambda$, the number of times it appears as a root of the characteristic equation must equal the number of linearly independent eigenvectors associated with it.
2. **Basis of Eigenvectors:** $A$ must possess $n$ linearly independent eigenvectors that form a basis for $\mathbb{R}^n$.

### The Case of Defective Matrices

If a matrix lacks $n$ independent eigenvectors, it is called **defective**.
**Example:** $A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$.
Both eigenvalues are $\lambda = 1$, but only one eigenvector $\mathbf{v} = [1, 0]^T$ exists. This matrix cannot be eigendecomposed (though it can be put into Jordan Normal Form).

---

## 3. Spectral Theorem for Symmetric Matrices

In ML, we predominantly work with **Symmetric Matrices** (e.g., Covariance matrices, Hessian matrices, Graph Laplacians). For a real symmetric matrix $A = A^T$:

1. All eigenvalues are **real numbers**.
2. Eigenvectors corresponding to distinct eigenvalues are **orthogonal**.
3. $A$ is always diagonalizable.

### Orthogonal Diagonalization

For symmetric matrices, we can choose an **orthonormal basis** of eigenvectors, making $P$ an orthogonal matrix ($P^T = P^{-1}$). The decomposition simplifies to:


$$\boxed{A = PDP^T}$$

---

## 4. Spectral Expansion (The Sum of Rank-One Matrices)

A more powerful way to view $A = PDP^T$ is by expanding the product into a sum of outer products:


$$A = \sum_{i=1}^{n} \lambda_i \mathbf{v}_i \mathbf{v}_i^T$$

### Properties of the Expansion:

* **Rank-1 Components:** Each term $\mathbf{v}_i \mathbf{v}_i^T$ is an $n \times n$ matrix with Rank 1.
* **Projection Operators:** If $\mathbf{v}_i$ are orthonormal, then $P_i = \mathbf{v}_i \mathbf{v}_i^T$ is an **orthogonal projection matrix** onto the subspace spanned by $\mathbf{v}_i$.
* **Information Hierarchy:** By ordering $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n$, we can approximate $A$ by taking only the first $k$ terms. This is the foundation of **Lossy Compression**.

---

## 5. Comprehensive Worked Example (Symmetric Case)

**Matrix:** $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$

**1. Characteristics:**

* $\det(A - \lambda I) = (\lambda-3)(\lambda-1) = 0 \implies \lambda_1 = 3, \lambda_2 = 1$.
* For $\lambda_1 = 3$: $\mathbf{v}_1 = [1, 1]^T \to \mathbf{u}_1 = \frac{1}{\sqrt{2}}[1, 1]^T$.
* For $\lambda_2 = 1$: $\mathbf{v}_2 = [-1, 1]^T \to \mathbf{u}_2 = \frac{1}{\sqrt{2}}[-1, 1]^T$.

**2. Spectral Decomposition:**


$$A = 3 \underbrace{\begin{bmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{bmatrix}}_{\mathbf{u}_1 \mathbf{u}_1^T} + 1 \underbrace{\begin{bmatrix} 0.5 & -0.5 \\ -0.5 & 0.5 \end{bmatrix}}_{\mathbf{u}_2 \mathbf{u}_2^T} = \begin{bmatrix} 1.5+0.5 & 1.5-0.5 \\ 1.5-0.5 & 1.5+0.5 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$

---

## 6. Functional Calculus and Matrix Powers

Spectral decomposition allows us to apply functions to matrices easily. If $A = PDP^{-1}$, then:

* **Matrix Powers:** $A^k = P D^k P^{-1}$. (Instead of multiplying $A$ a thousand times, we just raise the diagonal elements to the power of 1000).
* **Matrix Square Root:** $A^{1/2} = P D^{1/2} P^{-1}$ (Valid for Positive Semi-Definite matrices).
* **Matrix Exponential:** $e^A = P e^D P^{-1}$.

---

## 7. Importance in Machine Learning

* **PCA (Principal Component Analysis):** PCA is essentially the spectral decomposition of the covariance matrix $\Sigma = \frac{1}{m} X^T X$. The eigenvectors define the principal axes, and eigenvalues define the variance.
* **Spectral Clustering:** Uses the eigenvectors of the Graph Laplacian to map data into a space where clusters are easily separable by K-means.
* **Optimization:** The eigenvalues of the **Hessian matrix** $H$ determine the curvature of the loss surface. The condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ dictates the convergence speed of Gradient Descent.
* **Image Processing:** Used in transform coding (like the Karhunen–Loève transform) to decorrelate pixel data.

---
