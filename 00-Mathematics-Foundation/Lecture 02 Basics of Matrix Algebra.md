## Matrix Algebra for Machine Learning

*Essential Mathematics — Structured Notes*

---

## 1. Matrix Fundamentals

A matrix is a two-dimensional array of scalars (real numbers). It serves as the primary data structure for representing datasets and linear transformations in machine learning.

### Notation and Dimensions

* An $m \times n$ matrix consists of **m rows** and **n columns**.
* The element located at row $i$ and column $j$ is denoted as $a_{ij}$.
* **Square Matrix:** A matrix where $m = n$.
* **Rectangular Matrix:** A matrix where $m \neq n$.
* **Vectors as Matrices:** A column vector is an $m \times 1$ matrix, and a row vector is a $1 \times n$ matrix.

---

## 2. Special Matrix Architectures

Specific matrix structures simplify computations in optimization and dimensionality reduction:

* **Diagonal Matrix:** A square matrix where all entries outside the main diagonal are zero ($a_{ij} = 0$ for $i \neq j$).
* **Identity Matrix ($I_n$):** A diagonal matrix where all diagonal elements are 1. It acts as the multiplicative identity ($AI = A$).
* **Zero Matrix:** A matrix where every entry is 0.
* **Triangular Matrices:**
* **Upper Triangular:** Entries below the main diagonal are zero.
* **Lower Triangular:** Entries above the main diagonal are zero.



---

## 3. Basic Matrix Arithmetic

### Addition and Subtraction

Defined only for matrices of the same dimensions. Operations are performed component-wise:


$$(A \pm B)_{ij} = a_{ij} \pm b_{ij}$$

* **Commutative:** $A + B = B + A$
* **Associative:** $A + (B + C) = (A + B) + C$

### Scalar Multiplication

Multiplying a matrix by a scalar $\alpha$ scales every individual entry:


$$(\alpha A)_{ij} = \alpha \cdot a_{ij}$$

---

## 4. Matrix Multiplication

Matrix multiplication $C = AB$ is defined if the number of columns in $A$ equals the number of rows in $B$. If $A$ is $m \times n$ and $B$ is $n \times p$, then $C$ is $m \times p$.

### Computation Rule

The entry $c_{ij}$ is the dot product of the $i$-th row of $A$ and the $j$-th column of $B$:


$$c_{ij} = \sum_{k=1}^{n} a_{ik}b_{kj}$$

### Critical Properties

* **Non-Commutative:** Generally, $AB \neq BA$.
* **Associative:** $A(BC) = (AB)C$.
* **Distributive:** $A(B + C) = AB + AC$.
* **Null Product:** $AB = 0$ does not imply $A=0$ or $B=0$.

---

## 5. Transpose and Inverse

### Matrix Transpose ($A^T$)

The transpose is obtained by interchanging rows and columns. If $A$ is $m \times n$, $A^T$ is $n \times m$.

* $(AB)^T = B^T A^T$
* $(A+B)^T = A^T + B^T$

### Matrix Inverse ($A^{-1}$)

For a square matrix $A$, the inverse exists if and only if the determinant $\det(A) \neq 0$ (non-singular).

* $AA^{-1} = A^{-1}A = I$
* $(AB)^{-1} = B^{-1}A^{-1}$
* $(A^T)^{-1} = (A^{-1})^T$

---

## 6. Determinants and Adjugates

### Determinant

A scalar value representing the scaling factor of the linear transformation described by the matrix. For a $2 \times 2$ matrix:


$$\det \begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

### Adjugate Matrix ($adj(A)$)

The adjugate is the transpose of the cofactor matrix. It is used to calculate the inverse:


$$A^{-1} = \frac{1}{\det(A)} adj(A)$$


**Fundamental Identity:** $A \cdot adj(A) = \det(A)I$.

---

## 7. Orthogonal Matrices ($Q$)

A square matrix is orthogonal if its columns and rows are orthonormal vectors.

* **Definition:** $Q^T Q = QQ^T = I$
* **Implication:** $Q^{-1} = Q^T$
* **Properties:** Preserves vector lengths and angles; $\det(Q) = \pm 1$.

---

## 8. Computational Implementation (NumPy)

```python
import numpy as np

# Matrix definition
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Operations
addition = np.add(A, B)
product = np.matmul(A, B)      # or A @ B
transpose = A.T

# Linear Algebra Module
determinant = np.linalg.det(A)
inverse = np.linalg.inv(A)

```

---

## 9. Significance in Machine Learning

Matrix algebra provides the mathematical framework for nearly all ML algorithms:

* **Data Representation:** Feature matrices where rows are samples and columns are features.
* **Neural Networks:** Weight matrices define the connections between layers.
* **Optimization:** Gradient descent involves matrix-vector operations.
* **Dimensionality Reduction:** Techniques like PCA and SVD rely on eigendecomposition and singular value decomposition of matrices.

---
