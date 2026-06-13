## Eigenvalues and Eigenvectors

*Essential Mathematics for ML — Structured Notes*

---

## 1. Geometric Intuition

### Motivation
When a matrix multiplies a vector, it typically knocks it off its axis, twisting its direction entirely. But imagine a piece of stretchable rubber with a grid drawn on it. If you pull the rubber sheet outward from opposite sides, the entire grid deforms. However, the horizontal lines drawn precisely along the direction you are pulling *don't change their direction at all*—they only stretch perfectly parallel to themselves.

These un-twisting, rigid "bones" of a mathematical transformation are its **Eigenvectors**. The amount by which they stretched (or shrank) is the **Eigenvalue**. 

### Deep Learning Connection
In Recurrent Neural Networks (RNNs), a hidden state $\mathbf{h}$ is iteratively multiplied by the same weight matrix $W$ thousands of times over a sequence. If $W$ has an eigenvector with an eigenvalue $\lambda > 1$, multiplying it thousands of times will cause the vector to explode toward infinity ($1.1^{1000} \to \infty$). If $\lambda < 1$, it shrinks to nothing ($0.9^{1000} \to 0$). This is the literal mathematical origin of the **Vanishing / Exploding Gradient Problem**.

---

## 2. Mathematical Definition

Let $A$ be an $n \times n$ real square matrix. A nonzero vector $\mathbf{v}$ is an **eigenvector** if:

$$
A\mathbf{v} = \lambda \mathbf{v}
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $A$ | Square matrix transformation | Linear operator acting on vectors |
| $\mathbf{v}$ | Eigenvector (nonzero) | Direction preserved by transformation |
| $\lambda$ | Eigenvalue (scalar) | Factor by which eigenvector is stretched/shrunk |
| $A\mathbf{v}$ | Matrix-vector product | Result of applying transformation to eigenvector |
| $\lambda \mathbf{v}$ | Scalar multiplication | Equivalent effect to full matrix operation |

where $\lambda$ is a scalar called the **eigenvalue**. The matrix transformation acts on $\mathbf{v}$ exactly as if it were just a simple scalar multiplication.

> **Note:** The zero vector mathematically satisfies $A\mathbf{0} = \lambda \mathbf{0}$, but it provides no geometric information about the rigid rotational axes of the matrix, so it is strictly excluded from being an eigenvector.

---

## 3. How to Calculate Eigenvalues and Eigenvectors

We rearrange the definition into a homogeneous system:

$$
A\mathbf{v} - \lambda \mathbf{v} = \mathbf{0}
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $A\mathbf{v}$ | Matrix-vector product | Transformation result |
| $\lambda \mathbf{v}$ | Scalar multiple of eigenvector | Equivalent scalar effect |
| $\mathbf{0}$ | Zero vector | Target of homogeneous system |

$$
(A - \lambda I)\mathbf{v} = \mathbf{0}
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $A - \lambda I$ | Shifted matrix | Matrix adjusted by eigenvalue |
| $I$ | Identity matrix | Preserves vector space structure |
| $\mathbf{v}$ | Eigenvector | Nonzero solution to homogeneous system |
| $\mathbf{0}$ | Zero vector | Null space condition |

For a nonzero solution $\mathbf{v}$ to exist, the matrix $(A - \lambda I)$ must crush space to 0 (meaning it has a valid Null Space). It can only do that if its determinant is exactly zero. This is the **Characteristic Equation**:

$$
\det(A - \lambda I) = 0
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $\det$ | Determinant operator | Measures matrix singularity |
| $A - \lambda I$ | Shifted matrix | Must be singular for nonzero eigenvector |
| $\lambda$ | Eigenvalue | Root of the characteristic polynomial |
| $0$ | Zero determinant | Condition for nontrivial null space |

### Step-by-Step Procedure

1. Find the roots ($\lambda$) of the polynomial $\det(A - \lambda I) = 0$.
2. For each $\lambda$, plug it back into $(A - \lambda I)\mathbf{v} = \mathbf{0}$ and solve for the basis of the null space $\mathbf{v}$.

```python
import numpy as np

# A 3x3 Transformation Matrix
A = np.array([[2, -2,  3], 
              [1,  1,  1], 
              [1,  3, -1]])

# NumPy instantly solves the characteristic polynomial
values, vectors = np.linalg.eig(A)

print("Eigenvalues (The Stretching Factors):", values) 
# Output approx: [3., 1., -2.]

# The corresponding eigenvector for Eigenvalue 3
print("Eigenvector for Lambda=3:", vectors[:, 0])
```

---

## 4. Comprehensive Worked Example

**Matrix:** 

$$
A = \begin{bmatrix} 2 & -2 & 3 \\ 1 & 1 & 1 \\ 1 & 3 & -1 \end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $A$ | Square matrix being analyzed | $3 \times 3$ transformation whose eigenvalues and eigenvectors are sought |
| $2, -2, 3, \dots$ | Entries of $A$ | Coefficients defining the linear transformation |

**Step 1: Eigenvalues**
Solving $\det(A - \lambda I) = 0$ results in polynomial roots:

$$
(\lambda - 3)(\lambda - 1)(\lambda + 2) = 0
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $\lambda - 3$ | Factor for eigenvalue 3 | Root indicating stretch by factor 3 |
| $\lambda - 1$ | Factor for eigenvalue 1 | Root indicating no scaling |
| $\lambda + 2$ | Factor for eigenvalue -2 | Root indicating reflection and stretch |
| $= 0$ | Zero product | Each factor yields an eigenvalue |

**Eigenvalues:** $\lambda_1 = 3, \lambda_2 = 1, \lambda_3 = -2$.

**Step 2: Eigenvector for $\lambda_1 = 3$**
Substitute $\lambda = 3$ into $(A - 3I)\mathbf{v} = \mathbf{0}$:

$$
\begin{bmatrix} -1 & -2 & 3 \\ 1 & -2 & 1 \\ 1 & 3 & -4 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\begin{bmatrix} -1 & -2 & 3 \\ 1 & -2 & 1 \\ 1 & 3 & -4 \end{bmatrix}$ | Shifted matrix $A - 3I$ for $\lambda = 3$ | Must be singular for a non-trivial eigenvector to exist |
| $v_1, v_2, v_3$ | Components of the eigenvector | Unknown values to be solved from the null space |
| $\begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$ | Zero vector | Right-hand side of the homogeneous system |

Applying Gaussian elimination reveals $v_1 = v_2 = v_3$.
**Basis Eigenvector:** 

$$
\mathbf{v} = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{v}$ | Eigenvector for $\lambda = 3$ | Direction preserved by $A$; satisfies $A\mathbf{v} = 3\mathbf{v}$ |
| $1, 1, 1$ | Components of the eigenvector | All components equal, indicating equal contribution from each dimension |

---

## 5. Critical Matrix Properties

Knowing the eigenvalues allows you to "cheat" at advanced matrix algebra without actually doing the heavy computation:

* **Trace Property:** $\sum \lambda_i = \text{Tr}(A)$. (Sum of eigenvalues = sum of diagonal elements).
* **Determinant Property:** $\prod \lambda_i = \det(A)$. (Product of eigenvalues = determinant).
* **Matrix Inverse:** If a matrix is invertible, its inverse $A^{-1}$ holds the exact same eigenvectors, but its eigenvalues are explicitly $1/\lambda$.
* **Matrix Powers:** $A^{100}$ holds the exact same eigenvectors as $A$, but its eigenvalues are $\lambda^{100}$.

*Failure mode recap:* If *any* eigenvalue is exactly $0$, then the product of eigenvalues is $0$, meaning $\det(A) = 0$. The matrix has crushed a dimension and cannot be mathematically inverted.

---

## 6. Beyond the Basics: ML Applications

Eigenvalues are not just for solving textbooks; they are the core engine of almost every dimension reduction technique:

* **[Principal Component Analysis (PCA):](Lecture%2016%20Principal%20Component%20Analysis.md)** We find the directions of maximum variance by calculating the eigenvectors of the data's covariance matrix.
* **[Linear Discriminant Analysis (LDA):](Lecture%2019%20Linear%20Discriminant%20Analysis.md)** We separate data classes by finding the eigenvectors of $S_W^{-1} S_B$.
* **[Spectral Decomposition:](Lecture%2011%20Spectral%20decomposition.md)** Understanding how symmetric matrices can be "diagonalized" into their constituent components.

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 08: Orthogonal](Lecture%2008%20Orthogonal%20Complement%20and%20Projection%20Mapping.md) — Orthogonal projections and inner product spaces motivate the search for invariant directions
- **Next:** [Lecture 10: Special Matrices](Lecture%2010%20Special%20Matrices%20and%20Properties.md) — Classifies matrices (symmetric, orthogonal, PD) whose eigenvalues have special guarantees
- **Related:** [Lecture 11: Spectral](Lecture%2011%20Spectral%20decomposition.md) — Decomposes a matrix into its eigenvalue-weighted rank-1 components
- **Related:** [Lecture 16: PCA](Lecture%2016%20Principal%20Component%20Analysis.md) — Uses eigenvectors of the covariance matrix to find directions of maximum variance
- **Related:** [Lecture 19: LDA](Lecture%2019%20Linear%20Discriminant%20Analysis.md) — Uses generalized eigenvalues to find class-separating projections
