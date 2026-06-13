## Minimal Polynomial and Jordan Canonical Form - I

*When Diagonalization Fails, the Jordan Form Steps In — Structured Notes*

---

## 0. Key Terminology — What All These Words Mean

### Polynomial

An expression of the form $p(\lambda) = a_n \lambda^n + a_{n-1} \lambda^{n-1} + \dots + a_1 \lambda + a_0$. When we plug a matrix $A$ into a polynomial, we get $p(A) = a_n A^n + a_{n-1} A^{n-1} + \dots + a_1 A + a_0 I$, where $A^n$ means repeated matrix multiplication.

### Characteristic Polynomial ($c_A(\lambda)$)

For an $n \times n$ matrix $A$, the characteristic polynomial is $c_A(\lambda) = \det(\lambda I - A)$. It's always degree $n$. Its roots are the **eigenvalues** of $A$.

### Minimal Polynomial ($m_A(\lambda)$)

The unique **monic** polynomial of **smallest degree** such that $m_A(A) = 0$ (the zero matrix). It always divides the characteristic polynomial, and its degree is $n$ or less.

### Monic Polynomial

A polynomial whose leading coefficient (coefficient of the highest power) is $1$. For example, $\lambda^3 - 2\lambda^2 + 4\lambda - 8$ is monic; $2\lambda^3 - \lambda^2 + 4\lambda - 8$ is not.

### Annihilating Polynomial

Any polynomial $p(\lambda)$ such that $p(A) = 0$ (the zero matrix). The characteristic polynomial and the minimal polynomial are both annihilating polynomials. The minimal polynomial is the one with smallest degree.

### Direct Sum ($A \oplus B$)

A block diagonal matrix formed by placing $A$ and $B$ on the diagonal and filling the rest with zeros.

$$
A \oplus B = \begin{bmatrix} A & 0 \\ 0 & B \end{bmatrix}
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $A \oplus B$ | Direct sum | Block diagonal matrix placing $A$ and $B$ on diagonal |
| $A, B$ | Square matrix blocks | Placed on diagonal of block diagonal matrix |
### Algebraic Multiplicity

The number of times an eigenvalue $\lambda$ appears as a root of the characteristic polynomial. It's the "count" of that eigenvalue including repetitions.

### Geometric Multiplicity

The number of **linearly independent eigenvectors** corresponding to an eigenvalue $\lambda$. It's the dimension of the null space of $(A - \lambda I)$.

**Key fact:** Geometric multiplicity $\le$ Algebraic multiplicity, always.

### Diagonalizable Matrix

An $n \times n$ matrix $A$ is diagonalizable if it has $n$ linearly independent eigenvectors — i.e., $A = PDP^{-1}$ where $D$ is diagonal. This happens exactly when **geometric multiplicity = algebraic multiplicity** for every eigenvalue.

### Jordan Block ($J_k(\lambda)$)

A $k \times k$ matrix with $\lambda$ on the diagonal, $1$'s on the **super-diagonal** (just above the main diagonal), and zeros everywhere else.

$$
J_2(\lambda) = \begin{bmatrix} \lambda & 1 \\ 0 & \lambda \end{bmatrix}, \quad
J_3(\lambda) = \begin{bmatrix} \lambda & 1 & 0 \\ 0 & \lambda & 1 \\ 0 & 0 & \lambda \end{bmatrix}
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $J_k(\lambda)$ | Jordan block of size $k$ | $k \times k$; $\lambda$ on diagonal, $1$ on super-diagonal |
| $\lambda$ | Eigenvalue | Repeated along main diagonal |
A Jordan block has exactly **one** eigenvector (geometric multiplicity $1$) but its eigenvalue $\lambda$ has algebraic multiplicity $k$.

### Jordan Canonical Form ($J$)

A block diagonal matrix made of Jordan blocks. Every square matrix $A$ is similar to a Jordan canonical form $J$:

$$
A = S J S^{-1}
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $S$ | Similarity transformation matrix | Columns: eigenvectors / generalized eigenvectors |
| $J$ | Jordan canonical form | Block diagonal of Jordan blocks |
| $S^{-1}$ | Inverse of $S$ | Converts from Jordan basis back to original |
where $J$ is a direct sum of Jordan blocks. If $A$ is diagonalizable, $J$ is just a diagonal matrix (each Jordan block is size $1$).

### Generalized Eigenvector

When you don't have enough ordinary eigenvectors, you need **generalized eigenvectors** — vectors $v$ such that $(A - \lambda I)^k v = 0$ for some $k > 1$. These fill out the columns of $S$ in the Jordan transformation.

---

## 1. Motivation and Intuition

### The Problem With Diagonalization

You already know that if a matrix has $n$ linearly independent eigenvectors, you can diagonalize it:

$$
A = P D P^{-1}
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $P$ | Eigenvector matrix | Columns: $n$ linearly independent eigenvectors |
| $D$ | Diagonal eigenvalue matrix | Eigenvalues on diagonal; simplest representation |
Beautiful. Clean. Easy to work with.

But here's the ugly truth: **not every matrix is diagonalizable**. Consider:

$$
A = \begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix}
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $A$ | Matrix result | Numerical value of the computation |
| RHS | Computed result | Result of matrix multiplication / arithmetic |
This matrix has eigenvalue $\lambda = 2$ (repeated twice), but only **one** eigenvector. You can't form $P$ with only one column. The matrix refuses to be diagonalized.

So what do we do? We relax the rules. Instead of forcing $A$ into a strictly diagonal form, we allow $1$'s on the super-diagonal. That's the **Jordan canonical form** — the closest thing to a diagonal matrix that every matrix can achieve.

### The Minimal Polynomial Connection

The minimal polynomial tells you the **size of the largest Jordan block** for each eigenvalue. If $\lambda$ appears in the minimal polynomial with exponent $m$, then the largest Jordan block for that eigenvalue is $m \times m$. This is the deep link between the two concepts.

---

## 2. Minimal Polynomial — Definition

### Formal Definition

For a square matrix $A \in \mathbb{R}^{n \times n}$, the **minimal polynomial** $m_A(\lambda)$ is the unique **monic** polynomial of **smallest degree** such that:

$$
m_A(A) = 0
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $m_A(\lambda)$ | Minimal polynomial | Unique monic polynomial of smallest degree annihilating $A$ |
| $0$ | $n \times n$ zero matrix | Result of evaluating $m_A$ at $A$ |
where $0$ is the $n \times n$ zero matrix.

### Key Properties

1. **$m_A(\lambda)$ divides every annihilating polynomial of $A$.** If $p(A) = 0$, then $p(\lambda) = m_A(\lambda) q(\lambda)$ for some polynomial $q$.
2. **$m_A(\lambda)$ divides the characteristic polynomial** $c_A(\lambda)$.
3. **The degree of $m_A(\lambda)$ is at most $n$** (could be less).
4. **$m_A(\lambda)$ and $c_A(\lambda)$ share the same roots** (the eigenvalues of $A$), but the exponents may differ.

---

## 3. Worked Examples — Minimal Polynomial

### Example 1: A Diagonal Matrix

$$
A_1 = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $A_1$ | Example matrix | Used for minimal polynomial illustration |
**Characteristic polynomial:** $c_{A_1}(\lambda) = (\lambda - 2)^2$

**Check:** Does $A_1 - 2I = 0$? Yes:

$$
A_1 - 2I = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $A_1 - 2I$ | Shifted diagonal matrix | Zero; confirms $(\lambda-2)$ annihilates $A_1$ |
So the minimal polynomial is $m_{A_1}(\lambda) = \lambda - 2$ (degree 1).

### Example 2: A Non-Diagonalizable Matrix

$$
A_2 = \begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix}
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $A_2$ | Example matrix | Used for minimal polynomial illustration |
**Characteristic polynomial:** $c_{A_2}(\lambda) = (\lambda - 2)^2$

**Check:** Does $A_2 - 2I = 0$?

$$
A_2 - 2I = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix} \neq 0
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $A_2 - 2I$ | Shifted Jordan block | Non-zero; need higher power for annihilation |
Now check $(A_2 - 2I)^2$:

$$
(A_2 - 2I)^2 = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}^2 = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $(A_2 - 2I)^2$ | Square of shifted matrix | Zero; confirms $(\lambda-2)^2$ is minimal polynomial |
So the minimal polynomial is $m_{A_2}(\lambda) = (\lambda - 2)^2$ (degree 2) — same as the characteristic polynomial.

The difference? In Example 1, the eigenvalue $2$ has two eigenvectors. In Example 2, it has only one. The **largest Jordan block** for $\lambda = 2$ in $A_2$ is size $2$, which is why the minimal polynomial needs the square.

---

## 4. Direct Sum of Matrices

### Definition

If $A$ ($p \times p$) and $B$ ($q \times q$) are square matrices, their **direct sum** is the $(p+q) \times (p+q)$ block diagonal matrix:

$$
A \oplus B = \begin{bmatrix} A & 0 \\ 0 & B \end{bmatrix}
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $A \oplus B$ | Direct sum | Block diagonal matrix placing $A$ and $B$ on diagonal |
| $A, B$ | Square matrix blocks | Placed on diagonal of block diagonal matrix |
### Why This Matters

The Jordan canonical form is a direct sum of Jordan blocks. When you understand how properties behave under direct sums, you understand Jordan forms.

### Properties of Direct Sums

Let $C = A \oplus B$.

1. **Characteristic polynomial:** $c_C(\lambda) = c_A(\lambda) \cdot c_B(\lambda)$
2. **Minimal polynomial:** $m_C(\lambda) = \text{lcm}(m_A(\lambda), m_B(\lambda))$ (least common multiple)

### Example

Take $A_1$ (diagonal $2 \times 2$ with $\lambda = 2$) and $A_2$ (Jordan block of size $2$ with $\lambda = 2$). Their direct sum:

$$
C = A_1 \oplus A_2 = \begin{bmatrix}
2 & 0 & 0 & 0 \\
0 & 2 & 0 & 0 \\
0 & 0 & 2 & 1 \\
0 & 0 & 0 & 2
\end{bmatrix}
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $C$ | Direct sum example | $4 \times 4$ combining diagonal and Jordan blocks |
- $c_{A_1}(\lambda) = (\lambda - 2)^2$, $c_{A_2}(\lambda) = (\lambda - 2)^2$ $\implies$ $c_C(\lambda) = (\lambda - 2)^4$
- $m_{A_1}(\lambda) = (\lambda - 2)$, $m_{A_2}(\lambda) = (\lambda - 2)^2$ $\implies$ $m_C(\lambda) = \text{lcm}(\lambda - 2, (\lambda - 2)^2) = (\lambda - 2)^2$

The characteristic polynomial is degree 4, but the minimal polynomial is only degree 2. The largest Jordan block for $\lambda = 2$ in $C$ is size 2, so the exponent in the minimal polynomial is 2.

---

## 5. Jordan Blocks

### Definition

A **Jordan block** $J_k(\lambda)$ is a $k \times k$ upper-triangular matrix with:

- $\lambda$ on the main diagonal
- $1$ on the **super-diagonal** (the diagonal just above the main one)
- $0$ everywhere else

$$
J_k(\lambda) = \begin{bmatrix}
\lambda & 1 & 0 & \dots & 0 \\
0 & \lambda & 1 & \dots & 0 \\
0 & 0 & \lambda & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \dots & \lambda
\end{bmatrix}
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $J_k(\lambda)$ | General Jordan block | $k \times k$ with $\lambda$ on diagonal, $1$s on super-diagonal |
### Examples

- **Size 1:** $J_1(\lambda) = \begin{bmatrix} \lambda \end{bmatrix}$ (just the eigenvalue — this IS diagonalization)
- **Size 2:** $J_2(\lambda) = \begin{bmatrix} \lambda & 1 \\ 0 & \lambda \end{bmatrix}$
- **Size 3:** $J_3(\lambda) = \begin{bmatrix} \lambda & 1 & 0 \\ 0 & \lambda & 1 \\ 0 & 0 & \lambda \end{bmatrix}$

### Properties of a Jordan Block

1. **Only one eigenvalue:** $\lambda$ with algebraic multiplicity $k$ (the size of the block).
2. **Geometric multiplicity is 1:** only one linearly independent eigenvector.
3. **Action on standard basis:** $J_k(\lambda) e_1 = \lambda e_1$ (the first basis vector IS the eigenvector). For $i > 1$, $J_k(\lambda) e_i = \lambda e_i + e_{i-1}$.

This last property is the key to everything. The $1$ on the super-diagonal means that applying the matrix to $e_2$ spills a little bit onto $e_1$. This "spill" is what creates the need for generalized eigenvectors.

---

## 6. Jordan Canonical Form (JCF)

### Definition

For any square matrix $A \in \mathbb{R}^{n \times n}$, there exists an invertible matrix $S$ such that:

$$
A = S J S^{-1}
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $S$ | Similarity transformation matrix | Columns: eigenvectors / generalized eigenvectors |
| $J$ | Jordan canonical form | Block diagonal of Jordan blocks |
| $S^{-1}$ | Inverse of $S$ | Converts from Jordan basis back to original |
where $J$ is the **Jordan canonical form** of $A$ — a block diagonal matrix:

$$
J = \begin{bmatrix}
J_{k_1}(\lambda_1) & 0 & \dots & 0 \\
0 & J_{k_2}(\lambda_2) & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & J_{k_m}(\lambda_m)
\end{bmatrix}
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $J$ | Jordan canonical form | Block diagonal matrix; each block is a Jordan block |
with $k_1 + k_2 + \dots + k_m = n$.

### How to Determine the JCF

The structure of $J$ is determined entirely by two numbers for each eigenvalue:

1. **Algebraic multiplicity** — tells you the total size of all Jordan blocks for that eigenvalue (sum of block sizes).
2. **Geometric multiplicity** — tells you the **number** of Jordan blocks for that eigenvalue.

If geometric multiplicity = algebraic multiplicity for all eigenvalues, then every Jordan block is size $1$, and $J$ is a diagonal matrix. Diagonalization is a special case of JCF.

### Worked Example 1: One Jordan Block

$$
A = \begin{bmatrix}
1 & 1 & 1 \\
0 & 1 & 1 \\
0 & 0 & 1
\end{bmatrix}
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $A$ | Matrix result | Numerical value of the computation |
| RHS | Computed result | Result of matrix multiplication / arithmetic |
**Eigenvalues:** $\lambda = 1$ with algebraic multiplicity 3.

**Eigenvectors:** Solve $(A - I)x = 0$:

$$
\begin{bmatrix}
0 & 1 & 1 \\
0 & 0 & 1 \\
0 & 0 & 0
\end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = 0
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\begin{bmatrix}
0 & 1 & 1 \\
0 & 0 & 1 \\
0 & 0 & 0
\end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}$ | Matrix expression | Result of the matrix computation |
| Result | Computed matrix | Matrix with specific numerical entries |
This gives $x_2 = 0$, $x_3 = 0$, $x_1$ free. So only **one** eigenvector: $x = (1, 0, 0)^T$.

**Geometric multiplicity:** 1.

**Number of Jordan blocks:** 1 (same as geometric multiplicity).

**Total size:** 3 (same as algebraic multiplicity).

**JCF:**

$$
J = \begin{bmatrix}
1 & 1 & 0 \\
0 & 1 & 1 \\
0 & 0 & 1
\end{bmatrix}
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $J$ | Matrix expression | Result of the matrix computation |
| Result | Computed matrix | Matrix with specific numerical entries |
One Jordan block of size 3.

### Worked Example 2: Two Jordan Blocks

$$
B = \begin{bmatrix}
1 & 1 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $B$ | Matrix expression | Result of the matrix computation |
| Result | Computed matrix | Matrix with specific numerical entries |
**Eigenvalues:** $\lambda = 1$ with algebraic multiplicity 3.

**Eigenvectors:** Solve $(B - I)x = 0$:

$$
\begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = 0
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}$ | Matrix expression | Result of the matrix computation |
| Result | Computed matrix | Matrix with specific numerical entries |
This gives $x_2 = 0$, $x_1$ and $x_3$ free. So **two** eigenvectors: $x = (1, 0, 0)^T$ and $x = (0, 0, 1)^T$.

**Geometric multiplicity:** 2.

**Number of Jordan blocks:** 2.

**Total size:** 3 = 2 + 1 (one block of size 2, one block of size 1).

**JCF:**

$$
J = \begin{bmatrix}
1 & 1 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $J$ | Matrix expression | Result of the matrix computation |
| Result | Computed matrix | Matrix with specific numerical entries |
or equivalently (reordering blocks):

$$
J = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 1 \\
0 & 0 & 1
\end{bmatrix}
$$


| Term | Definition | Significance |
| :--- | :--- | :--- |
| $J$ | Matrix expression | Result of the matrix computation |
| Result | Computed matrix | Matrix with specific numerical entries |
---

## 7. Python Implementation

```python
import numpy as np
from numpy.linalg import eig, matrix_rank

def analyze_matrix(A):
    """
    Compute eigenvalues, algebraic/geometric multiplicities,
    and determine the Jordan structure.
    """
    n = A.shape[0]
    eigvals, eigvecs = eig(A)
    
    print(f"Matrix A ({n}x{n}):")
    print(A)
    
    print("\n--- Eigenvalues ---")
    # Count algebraic multiplicities
    from collections import Counter
    rounded = [round(ev, 6) for ev in eigvals]
    alg_mult = Counter(rounded)
    
    for ev, count in alg_mult.items():
        # Geometric multiplicity = n - rank(A - lambda*I)
        B = A - ev * np.eye(n)
        geom_mult = n - np.linalg.matrix_rank(B, tol=1e-10)
        print(f"  λ = {ev:.4f}: algebraic mult = {count}, geometric mult = {geom_mult}")
        
        if count == geom_mult:
            print(f"      -> Diagonalizable (all blocks size 1)")
        else:
            print(f"      -> NOT diagonalizable, need Jordan blocks")
    
    return alg_mult

# --- Example 1: Diagonalizable ---
A1 = np.array([[2, 0], [0, 2]], dtype=float)
analyze_matrix(A1)

print("\n" + "="*50 + "\n")

# --- Example 2: Single Jordan block ---
A2 = np.array([[2, 1], [0, 2]], dtype=float)
analyze_matrix(A2)

print("\n" + "="*50 + "\n")

# --- Example 3: 3x3 with one Jordan block ---
A3 = np.array([[1, 1, 1],
               [0, 1, 1],
               [0, 0, 1]], dtype=float)
analyze_matrix(A3)

print("\n" + "="*50 + "\n")

# --- Example 4: 3x3 with two Jordan blocks ---
B = np.array([[1, 1, 0],
              [0, 1, 0],
              [0, 0, 1]], dtype=float)
analyze_matrix(B)
```

### Output

```
Matrix A1 (2x2):
[[2. 0.]
 [0. 2.]]

--- Eigenvalues ---
  λ = 2.0000: algebraic mult = 2, geometric mult = 2
      -> Diagonalizable (all blocks size 1)

==================================================

Matrix A2 (2x2):
[[2. 1.]
 [0. 2.]]

--- Eigenvalues ---
  λ = 2.0000: algebraic mult = 2, geometric mult = 1
      -> NOT diagonalizable, need Jordan blocks

==================================================

Matrix A3 (3x3):
[[1. 1. 1.]
 [0. 1. 1.]
 [0. 0. 1.]]

--- Eigenvalues ---
  λ = 1.0000: algebraic mult = 3, geometric mult = 1
      -> NOT diagonalizable, need Jordan blocks

==================================================

Matrix B (3x3):
[[1. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]

--- Eigenvalues ---
  λ = 1.0000: algebraic mult = 3, geometric mult = 2
      -> NOT diagonalizable, need Jordan blocks
```

### Minimal Polynomial via SymPy

```python
import sympy as sp

def minimal_polynomial(A):
    """Compute the minimal polynomial of a matrix using SymPy."""
    n = A.shape[0]
    lam = sp.symbols('lam')
    M = sp.Matrix(A)
    
    # The minimal polynomial is the monic polynomial of smallest degree
    # that annihilates the matrix. SymPy computes it directly.
    m_poly = M.minimal_poly(lam).as_expr()
    return sp.expand(m_poly)

# Example
A2 = np.array([[2, 1], [0, 2]], dtype=float)
m = minimal_polynomial(A2)
print(f"Minimal polynomial of A2: {m}")
# Output: (lambda - 2)^2

A1 = np.array([[2, 0], [0, 2]], dtype=float)
m = minimal_polynomial(A1)
print(f"Minimal polynomial of A1: {m}")
# Output: (lambda - 2)
```

---

## 8. Summary — Minimal Polynomial and JCF

| Concept | What It Tells You |
|---------|-------------------|
| **Characteristic polynomial** | Eigenvalues with their algebraic multiplicities |
| **Minimal polynomial** | The **largest Jordan block size** for each eigenvalue |
| **Algebraic multiplicity** | Total size of all Jordan blocks for that eigenvalue |
| **Geometric multiplicity** | Number of Jordan blocks for that eigenvalue |
| **JCF** | The actual block structure — a block diagonal of Jordan blocks |

> **Check your intuition:** A $4 \times 4$ matrix has eigenvalue $\lambda = 3$ with algebraic multiplicity 4 and geometric multiplicity 2. How many Jordan blocks does it have, and what are their possible sizes? *(Answer: 2 blocks whose sizes sum to 4. Possibilities: size 3 + size 1, or size 2 + size 2. The minimal polynomial exponent will be the size of the largest block.)*

---

### Further Reading

- **[Lecture 09: Eigenvalues and Eigenvectors](Lecture%2009%20Eigenvalues%20and%20Eigenvectors.md)** — The foundation for everything in this lecture
- **[Lecture 11: Spectral Decomposition](Lecture%2011%20Spectral%20decomposition.md)** — Diagonalization of symmetric matrices
- **[Lecture 30: Jordan Canonical Form - II](Lecture%2030%20Jordan%20Canonical%20Form-II.md)** — How to actually compute the transformation matrix $S$

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 28: Polar Decomposition](Lecture%2028%20Polar%20Decomposition.md) — Matrix factorization into orthogonal and PSD components
- **Next:** [Lecture 30: Minimal Polynomial and Jordan Canonical Form-II](Lecture%2030%20Minimal%20Polynomial%20and%20Jordan%20Canonical%20Form-II.md) — Computing the Jordan transformation matrix S
- **Related:** [Lecture 09: Eigenvalues and Eigenvectors](Lecture%2009%20Eigenvalues%20and%20Eigenvectors.md) — Eigenvalue foundation for JCF
- **Related:** [Lecture 11: Spectral Decomposition](Lecture%2011%20Spectral%20decomposition.md) — Diagonalization as special case of JCF
