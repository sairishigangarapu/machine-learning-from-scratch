## 1. Geometric Motivation

When a matrix $A$ multiplies a vector $x$, it produces a new vector $y = Ax$. This transformation typically changes both the **magnitude** and the **direction** of $x$.

Example:

For $A = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}$ and $x = \begin{pmatrix} 1 \\ 3 \end{pmatrix}$:

$$Ax = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix} \begin{pmatrix} 1 \\ 3 \end{pmatrix} = \begin{pmatrix} 7 \\ 5 \end{pmatrix}$$

The resulting vector $(7, 5)$ has a different direction than $(1, 3)$.

However, for certain special vectors, the matrix multiplication only scales the vector without changing its direction.

For the same matrix $A$, if we choose $x = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$:

$$Ax = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix} \begin{pmatrix} 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 3 \\ 3 \end{pmatrix} = 3 \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

Here, the output is just $3$ times the input. The direction remains the same.

- **Eigenvector ($v$):** A non-zero vector whose direction does not change when a linear transformation is applied.
    
- **Eigenvalue ($\lambda$):** The scalar factor by which the eigenvector is scaled.
    

---

## 2. Mathematical Definition

Let $A$ be an $n \times n$ real square matrix. A non-zero vector $v \in \mathbb{R}^n$ is an **eigenvector** of $A$ corresponding to the **eigenvalue** $\lambda$ if:

$$A v = \lambda v$$

This implies:

$$(A - \lambda I) v = 0$$

where $I$ is the $n \times n$ identity matrix and $0$ is the zero vector.

---

## 3. Calculation of Eigenvalues and Eigenvectors

To find non-zero solutions for $v$ in the homogeneous system $(A - \lambda I)v = 0$, the matrix $(A - \lambda I)$ must be **rank deficient** (singular).

### Step 1: Characteristic Equation

Set the determinant to zero to solve for $\lambda$:

$$\det(A - \lambda I) = 0$$

This results in a polynomial of degree $n$ called the Characteristic Polynomial. The roots of this polynomial are the eigenvalues.

### Step 2: Find Eigenvectors

For each $\lambda$, solve the system $(A - \lambda I)v = 0$ to find the corresponding vector $v$.

### Example

Given $A = \begin{pmatrix} 2 & -2 & 3 \\ 1 & 1 & 1 \\ 1 & 3 & -1 \end{pmatrix}$:

1. Find Eigenvalues:
    
    $$\det \begin{pmatrix} 2-\lambda & -2 & 3 \\ 1 & 1-\lambda & 1 \\ 1 & 3 & -1-\lambda \end{pmatrix} = 0$$
    
    Solving this cubic polynomial gives roots: $\lambda = 3, 1, -2$.
    
2. Find Eigenvector for $\lambda = 3$:
    
    Solve $(A - 3I)v = 0$:
    
    $$\begin{pmatrix} -1 & -2 & 3 \\ 1 & -2 & 1 \\ 1 & 3 & -4 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \\ v_3 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix}$$
    
    Solution: $v_1 = v_2 = v_3$. One solution is $v = \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}$.
    

---

## 4. Geometric Interpretation of Transformations

Eigenvalues characterize the transformation applied to an object:

- **Diagonal Matrix:** A matrix like $\begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}$ scales the x-axis by 2 and leaves the y-axis unchanged. The basis vectors are the eigenvectors.
    
- **Shear/Rotation:** If an object (like a square) is transformed into a parallelogram, the eigenvectors represent the axes along which the shape is merely stretched, not rotated.
    

---

## 5. Properties

1. Invertibility:
    
    If $Av = \lambda v$ and $A$ is invertible, then:
    
    $$A^{-1} v = \frac{1}{\lambda} v$$
    
    (The eigenvectors are the same, eigenvalues are reciprocals).
    
2. Powers:
    
    If $Av = \lambda v$, then for any integer $k$:
    
    $$A^k v = \lambda^k v$$
    
3. Trace:
    
    The sum of the diagonal elements (Trace) equals the sum of eigenvalues:
    
    $$\text{tr}(A) = \sum_{i=1}^n \lambda_i$$
    
4. Determinant:
    
    The determinant of $A$ equals the product of eigenvalues:
    
    $$\det(A) = \prod_{i=1}^n \lambda_i$$
    
    Note: If any $\lambda = 0$, the matrix is singular.
    
5. Algebraic Multiplicity:
    
    If a root of the characteristic polynomial repeats $k$ times, the algebraic multiplicity of that eigenvalue is $k$.
    
6. Symmetric Matrices:
    
    For real symmetric matrices, eigenvectors corresponding to distinct eigenvalues are always orthogonal.
    

---

## 6. Python Implementation (`numpy`)

We can use `numpy.linalg.eig` to compute these values.

Python

```python
import numpy as np

# Define the matrix A
A = np.array([[2, -2, 3], 
              [1, 1, 1], 
              [1, 3, -1]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Note: Numpy returns normalized eigenvectors (unit length).
# The columns of the 'eigenvectors' array correspond to the 'eigenvalues'.
```

### Diagonal Matrix Example

For a diagonal matrix, the eigenvalues are simply the diagonal elements.

Python

```python
D = np.array([[2, 0, 0],
              [0, 1, 0],
              [0, 0, -1]])
              
vals, vecs = np.linalg.eig(D)
# Output vals: [2., 1., -1.]
# Output vecs: Standard basis vectors (columns of Identity matrix)
```