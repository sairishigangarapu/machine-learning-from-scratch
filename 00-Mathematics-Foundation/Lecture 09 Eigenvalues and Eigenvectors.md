
## Eigenvalues and Eigenvectors

*Essential Mathematics for ML — Structured Notes*

---

## 1. Geometric Intuition

In a linear transformation, most vectors change both their length and their direction. However, for every square matrix, there exist special vectors that **do not change their direction** when the matrix hits them. They only undergo scaling.

* **Eigenvector:** A special nonzero vector that stays on its own span after transformation.
* **Eigenvalue ($\lambda$):** The scalar factor by which the eigenvector is stretched or squashed.

### The Linear Mapping Analogy

Consider a matrix $A$ hitting a vector $\mathbf{x}$. Usually, the output $\mathbf{y} = A\mathbf{x}$ points in a new direction. If $\mathbf{x}$ is an eigenvector, then:


$$A\mathbf{x} = \lambda \mathbf{x}$$


The direction remains constant; only the magnitude changes by a factor of $\lambda$.

---

## 2. Mathematical Definition

Let $A$ be an $n \times n$ real matrix. A nonzero vector $\mathbf{v} \in \mathbb{R}^n$ is an **eigenvector** of $A$ if:


$$A\mathbf{v} = \lambda \mathbf{v}$$


where $\lambda$ is a scalar called the **eigenvalue**.

> **Note:** Eigenvectors must be **nonzero** vectors. The zero vector always satisfies the equation but provides no information about the transformation's direction.

---

## 3. How to Calculate Eigenvalues and Eigenvectors

To find these pairs, we rearrange the definition into a homogeneous system:


$$(A - \lambda I)\mathbf{v} = \mathbf{0}$$


For a nonzero solution $\mathbf{v}$ to exist, the matrix $(A - \lambda I)$ must be singular (its rank must be less than $n$). This leads to the **Characteristic Equation**:

$$\det(A - \lambda I) = 0$$

### Step-by-Step Procedure

1. **Solve the Characteristic Equation:** Find the roots ($\lambda$) of the polynomial $\det(A - \lambda I) = 0$.
2. **Solve for Eigenspace:** For each $\lambda$, plug it back into $(A - \lambda I)\mathbf{v} = \mathbf{0}$ and solve for the components of $\mathbf{v}$.

---

## 4. Comprehensive Worked Example

**Matrix:** $A = \begin{bmatrix} 2 & -2 & 3 \\ 1 & 1 & 1 \\ 1 & 3 & -1 \end{bmatrix}$

**Step 1: Eigenvalues**
Solving $\det(A - \lambda I) = 0$ results in a 3rd-degree polynomial:


$$(\lambda - 3)(\lambda - 1)(\lambda + 2) = 0$$


**Eigenvalues:** $\lambda_1 = 3, \lambda_2 = 1, \lambda_3 = -2$.

**Step 2: Eigenvector for $\lambda_1 = 3$**
Substitute $\lambda = 3$ into $(A - 3I)\mathbf{v} = \mathbf{0}$:


$$\begin{bmatrix} -1 & -2 & 3 \\ 1 & -2 & 1 \\ 1 & 3 & -4 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$$


Solving this system yields $v_1 = v_2 = v_3$.
**Basis Eigenvector:** $\mathbf{v} = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}$.

---

## 5. Important Properties

### Matrix Relationships

* **Trace Property:** $\sum \lambda_i = \text{Tr}(A)$. The sum of all eigenvalues equals the sum of the diagonal elements of the matrix.
* **Determinant Property:** $\prod \lambda_i = \det(A)$. The product of all eigenvalues equals the determinant of the matrix.
* **Invertibility:** A matrix $A$ is invertible if and only if $0$ is **not** an eigenvalue of $A$. If $\det(A) = 0$, at least one eigenvalue must be zero.

### Transformations of Eigenvalues

* **Inverse Matrix:** If $\lambda$ is an eigenvalue of an invertible matrix $A$, then $1/\lambda$ is an eigenvalue of $A^{-1}$ with the same eigenvector.
* **Powers of a Matrix:** If $\lambda$ is an eigenvalue of $A$, then $\lambda^k$ is an eigenvalue of $A^k$ for any integer $k > 0$.
* **Scalar Multiples:** If $\lambda$ is an eigenvalue of $A$, then $c\lambda$ is an eigenvalue of $cA$.

---

## 6. Implementation in Python (NumPy)

```python
import numpy as np

# Define Matrix from the example
A = np.array([[2, -2, 3], 
              [1, 1, 1], 
              [1, 3, -1]])

# Calculate Eigenvalues and Eigenvectors
values, vectors = np.linalg.eig(A)

print("Eigenvalues:", values)

# Eigenvectors are returned as columns in a matrix
# To see the eigenvector for the first eigenvalue:
print("First Eigenvector:", vectors[:, 0])

```

---

## 7. Significance in Machine Learning

* **Dimensionality Reduction (PCA):** The principal components of a dataset are the eigenvectors of its covariance matrix. Eigenvalues quantify the variance captured in each direction.
* **Transformation Characterization:** Eigenvectors define the **direction of change**, while eigenvalues define the **amount of scaling** (magnitude) along that direction.
* **Spectral Clustering:** Uses the eigenvalues of the Laplacian matrix of a graph to perform dimensionality reduction before clustering in fewer dimensions.

---

*Note: You have **987** prompts remaining for today.*

**Would you like me to move on to Diagonalization or Singular Value Decomposition (SVD)?**
