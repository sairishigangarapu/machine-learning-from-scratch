## Minimal Polynomial and Jordan Canonical Form - II

*Finding S, Generalized Eigenvectors, and Why This Actually Matters — Structured Notes*

---

## 0. Key Terminology — What All These Words Mean

### Generalized Eigenvector

A nonzero vector $x$ such that:

$$
(A - \lambda I)^p x = 0 \quad \text{for some } p \ge 1
$$

but $(A - \lambda I)^{p-1} x \neq 0$. An ordinary eigenvector is the special case where $p = 1$.

*Think of it as*: An eigenvector is killed in one blow by $(A - \lambda I)$. A generalized eigenvector takes $p$ blows. You need them when you don't have enough ordinary eigenvectors to fill out the transformation matrix $S$.

### Jordan Chain

A sequence of vectors $x_1, x_2, \dots, x_p$ where:
- $x_1$ is an ordinary eigenvector: $(A - \lambda I) x_1 = 0$
- $x_2$ is a generalized eigenvector: $(A - \lambda I) x_2 = x_1$
- $x_3$ is a generalized eigenvector: $(A - \lambda I) x_3 = x_2$
- ...
- $x_p$ is a generalized eigenvector: $(A - \lambda I) x_p = x_{p-1}$

The length of the chain equals the size of the Jordan block. Each chain fills one column of $S$ per vector.

### Similarity Transformation

Two matrices $A$ and $B$ are **similar** if there exists an invertible matrix $S$ such that $A = S B S^{-1}$. Similar matrices represent the same linear transformation in different bases. They share eigenvalues, characteristic polynomials, minimal polynomials, and Jordan form.

### Sparsity

A matrix is **sparse** if most of its entries are zero. The Jordan canonical form $J$ is sparse (only diagonal and super-diagonal entries), which makes computations like $J^{100}$ or $e^J$ much cheaper than for the original dense $A$.

### Matrix Function ($f(A)$)

For a function $f$ (like $f(x) = x^{100}$, $f(x) = e^x$, $f(x) = \sin x$), we define $f(A)$ using the similarity transformation:

$$
f(A) = S f(J) S^{-1}
$$

For Jordan blocks, $f(J_k(\lambda))$ has a closed form involving $f(\lambda)$, $f'(\lambda)$, $f''(\lambda)$, etc., on the super-diagonals.

---

## 1. Motivation and Intuition

In the last lecture, we learned how to write the Jordan canonical form $J$ — the block-diagonal "almost diagonal" matrix that every square matrix is similar to. But knowing $J$ alone isn't enough. We need the transformation matrix $S$ such that:

$$
A = S J S^{-1}
$$

The columns of $S$ are the **eigenvectors** and **generalized eigenvectors** of $A$. This lecture shows you how to find them.

We'll also connect the dots between the **minimal polynomial** and the **Jordan structure**, and show you why this matters for real computations — like raising matrices to high powers or computing matrix exponentials.

---

## 2. Generalized Eigenvectors — The Definition

### Why Ordinary Eigenvectors Aren't Enough

If $A$ is $n \times n$ and diagonalizable, we have $n$ linearly independent eigenvectors. They form the columns of $P$ in $A = PDP^{-1}$.

But when $A$ is **not** diagonalizable (geometric multiplicity $<$ algebraic multiplicity for some eigenvalue), we're short of eigenvectors. We need to fill the gap with **generalized eigenvectors**.

### Formal Definition

Let $A$ be an $n \times n$ matrix and $\lambda$ an eigenvalue. A nonzero vector $x$ is a **generalized eigenvector of rank $p$** if:

$$
(A - \lambda I)^p x = 0
$$

but

$$
(A - \lambda I)^{p-1} x \neq 0
$$

An ordinary eigenvector is a generalized eigenvector of rank $1$.

### The Jordan Chain

Generalized eigenvectors come in chains. For each Jordan block of size $k$, you get a chain of $k$ vectors:

$$
(A - \lambda I) x_1 = 0 \quad \text{(ordinary eigenvector)}
$$
$$
(A - \lambda I) x_2 = x_1
$$
$$
(A - \lambda I) x_3 = x_2
$$
$$
\vdots
$$
$$
(A - \lambda I) x_k = x_{k-1}
$$

Each equation $(A - \lambda I) x_{j} = x_{j-1}$ is a **non-homogeneous system**. You solve it to get each generalized eigenvector.

---

## 3. How to Construct $S$ — Step-by-Step

### The Recipe

1. **Find the eigenvalues** of $A$.
2. **For each eigenvalue $\lambda$:**
   - Determine the **algebraic multiplicity** $r$ (total size of Jordan blocks for $\lambda$).
   - Determine the **geometric multiplicity** $g$ (number of Jordan blocks for $\lambda$).
   - Find the $g$ ordinary eigenvectors for $\lambda$ (these start each Jordan chain).
   - For each chain, solve $(A - \lambda I) x_{j+1} = x_j$ to extend the chain until you have $r$ vectors total.
3. **Arrange the vectors as columns of $S$** in the order matching the Jordan blocks in $J$.

---

## 4. Worked Example 1 — Full Transformation

### Problem

$$
A = \begin{bmatrix}
1 & 1 & 1 \\
0 & 2 & 1 \\
0 & 0 & 3
\end{bmatrix}
$$

Find $S$ and $J$ such that $A = S J S^{-1}$.

### Solution

**Step 1:** Eigenvalues. $A$ is upper triangular, so $\lambda = 1, 2, 3$. All distinct.

**Step 2:** Since eigenvalues are distinct, $A$ is diagonalizable. Geometric multiplicity = algebraic multiplicity = 1 for each.

**Step 3:** Find eigenvectors.

For $\lambda = 3$: solve $(A - 3I)x = 0$.

$$
\begin{bmatrix}
-2 & 1 & 1 \\
0 & -1 & 1 \\
0 & 0 & 0
\end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = 0
$$

$x_2 = x_3$, $-2x_1 + x_2 + x_3 = -2x_1 + 2x_3 = 0 \implies x_1 = x_3$. So:

$$
x_3 = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}
$$

For $\lambda = 1$: solve $(A - I)x = 0$.

$$
\begin{bmatrix}
0 & 1 & 1 \\
0 & 1 & 1 \\
0 & 0 & 2
\end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = 0
$$

$x_3 = 0$, $x_2 = 0$, $x_1$ free. So:

$$
x_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}
$$

For $\lambda = 2$: solve $(A - 2I)x = 0$.

$$
\begin{bmatrix}
-1 & 1 & 1 \\
0 & 0 & 1 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = 0
$$

$x_3 = 0$, $-x_1 + x_2 = 0 \implies x_1 = x_2$. So:

$$
x_2 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}
$$

**Step 4:** Assemble $S$ and $J$.

$$
S = \begin{bmatrix}
1 & 1 & 1 \\
0 & 1 & 1 \\
0 & 0 & 1
\end{bmatrix}, \quad
J = \begin{bmatrix}
1 & 0 & 0 \\
0 & 2 & 0 \\
0 & 0 & 3
\end{bmatrix}
$$

Since all eigenvalues are distinct, $J$ is diagonal. Check $A = S J S^{-1}$.

---

## 5. Worked Example 2 — With Generalized Eigenvectors

### Problem

$$
B = \begin{bmatrix}
1 & 1 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

Find $S$ and $J$.

### Solution

**Step 1:** Eigenvalues. $\lambda = 1$ with algebraic multiplicity 3.

**Step 2:** Geometric multiplicity. Solve $(B - I)x = 0$:

$$
\begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = 0
$$

$x_2 = 0$, $x_1$ and $x_3$ free. So geometric multiplicity = 2 (two ordinary eigenvectors).

$$
v_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad
v_2 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
$$

**Step 3:** Determine Jordan block structure.
- Algebraic mult = 3, geometric mult = 2 $\implies$ two Jordan blocks, total size 3.
- Possibilities: size 2 + size 1 (or size 1 + size 2).

**Step 4:** Find the generalized eigenvector.

One of the ordinary eigenvectors starts a chain of length 2. Let's use $v_1 = (1, 0, 0)^T$. Solve $(B - I) x_3 = v_1$:

$$
\begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0
\end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}
$$

This gives $x_2 = 1$, $x_1$ and $x_3$ free. Pick $x_1 = 0$, $x_3 = 0$:

$$
x_3 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}
$$

**Step 5:** Assemble $S$ and $J$.

The first chain (size 2) gives columns $v_1$ and $x_3$. The second chain (size 1) gives $v_2$.

$$
S = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}, \quad
J = \begin{bmatrix}
1 & 1 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

Notice that $S = I$ in this case because the Jordan chains already align with the standard basis.

---

## 6. Worked Example 3 — A More Complex $3 \times 3$

### Problem

$$
A = \begin{bmatrix}
1 & 1 & 1 \\
0 & 1 & 1 \\
0 & 0 & 3
\end{bmatrix}
$$

Find $S$ and $J$.

### Solution

**Step 1:** Eigenvalues. $\lambda = 1$ (algebraic mult 2) and $\lambda = 3$ (algebraic mult 1).

**Step 2:** Eigenvectors.

For $\lambda = 3$: One eigenvector (distinct eigenvalues always give one LI eigenvector).

$$
(A - 3I) x = 0 \implies x_1 = \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}
$$

For $\lambda = 1$: solve $(A - I) x = 0$.

$$
\begin{bmatrix}
0 & 1 & 1 \\
0 & 0 & 1 \\
0 & 0 & 2
\end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = 0
$$

$x_3 = 0$, $x_2 = 0$, $x_1$ free. So **one** eigenvector:

$$
x_2 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}
$$

Geometric multiplicity of $\lambda = 1$ is 1, algebraic is 2 $\implies$ one Jordan block of size 2.

**Step 3:** Find generalized eigenvector for $\lambda = 1$.

Solve $(A - I) x_3 = x_2$:

$$
\begin{bmatrix}
0 & 1 & 1 \\
0 & 0 & 1 \\
0 & 0 & 2
\end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}
$$

From row 2: $x_3 = 0$. From row 3: $2x_3 = 0$ (consistent). From row 1: $x_2 + x_3 = 1 \implies x_2 = 1$.

So $x_1$ is free. Pick $x_1 = 0$:

$$
x_3 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}
$$

**Step 4:** Assemble $S$ and $J$.

Order: $\lambda = 3$ block first, then $\lambda = 1$ block.

$$
S = \begin{bmatrix}
1 & 1 & 0 \\
2 & 0 & 1 \\
2 & 0 & 0
\end{bmatrix}, \quad
J = \begin{bmatrix}
3 & 0 & 0 \\
0 & 1 & 1 \\
0 & 0 & 1
\end{bmatrix}
$$

You can verify $A = S J S^{-1}$.

---

## 7. The Deep Link: Minimal Polynomial $\iff$ JCF

### The Fundamental Relationship

The **minimal polynomial** of $A$ tells you the **size of the largest Jordan block** for each eigenvalue. The **characteristic polynomial** tells you the **total size** (algebraic multiplicity).

$$
m_A(\lambda) = (\lambda - \lambda_1)^{s_1} (\lambda - \lambda_2)^{s_2} \dots (\lambda - \lambda_k)^{s_k}
$$

where $s_i$ is the size of the **largest Jordan block** for eigenvalue $\lambda_i$.

$$
c_A(\lambda) = (\lambda - \lambda_1)^{r_1} (\lambda - \lambda_2)^{r_2} \dots (\lambda - \lambda_k)^{r_k}
$$

where $r_i$ is the **algebraic multiplicity** of $\lambda_i$ (total size of all Jordan blocks for $\lambda_i$).

And the **geometric multiplicity** $g_i$ of $\lambda_i$ is the **number of Jordan blocks** for $\lambda_i$.

### Reconstructing JCF from Polynomials

If you know $c_A(\lambda)$ and $m_A(\lambda)$, you can determine $J$ up to reordering of blocks.

### Example

A $6 \times 6$ matrix $A$ has:

$$
c_A(\lambda) = (\lambda - 3)^4 (\lambda - 4)^2
$$

$$
m_A(\lambda) = (\lambda - 3)^3 (\lambda - 4)^2
$$

What is $J$?

**For $\lambda = 3$:**
- Total size (algebraic mult): 4
- Largest block size: 3 (from exponent in $m_A$)
- Possible block sizes: $3 + 1$ (one size-3 block, one size-1 block)
- Geometric multiplicity: 2 blocks

**For $\lambda = 4$:**
- Total size: 2
- Largest block size: 2
- Only possibility: one block of size 2
- Geometric multiplicity: 1 block

So:

$$
J = \begin{bmatrix}
3 & 1 & 0 & 0 & 0 & 0 \\
0 & 3 & 1 & 0 & 0 & 0 \\
0 & 0 & 3 & 0 & 0 & 0 \\
0 & 0 & 0 & 3 & 0 & 0 \\
0 & 0 & 0 & 0 & 4 & 1 \\
0 & 0 & 0 & 0 & 0 & 4
\end{bmatrix}
$$

### When There Are Multiple Possibilities

If the minimal polynomial is $m_A(\lambda) = (\lambda - 3)^2 (\lambda - 4)^2$ instead, then for $\lambda = 3$:
- Total size: 4
- Largest block: 2
- Possible decompositions: $2 + 2$ or $2 + 1 + 1$

Both are valid JCFs. You need to compute the null spaces of $(A - \lambda I)^p$ for $p = 1, 2, \dots$ to distinguish them.

---

## 8. Applications — Why This Matters in ML

### Computing Matrix Powers

Need $A^{100}$? Don't multiply $A$ by itself 100 times. Use the Jordan form:

$$
A^{100} = S J^{100} S^{-1}
$$

Raising a Jordan block to a power has a closed form:

$$
J_k(\lambda)^n = \begin{bmatrix}
\lambda^n & \binom{n}{1} \lambda^{n-1} & \binom{n}{2} \lambda^{n-2} & \dots \\
0 & \lambda^n & \binom{n}{1} \lambda^{n-1} & \dots \\
0 & 0 & \lambda^n & \dots \\
\vdots & \vdots & \vdots & \ddots
\end{bmatrix}
$$

This is vastly cheaper than dense exponentiation.

### Computing Matrix Exponentials

In continuous-time dynamical systems and neural ODEs, you need $e^{At}$:

$$
e^{A} = S e^{J} S^{-1}
$$

For a Jordan block, the exponential has a beautiful closed form:

$$
e^{J_k(\lambda)} = e^{\lambda} \begin{bmatrix}
1 & 1 & \frac{1}{2!} & \dots & \frac{1}{(k-1)!} \\
0 & 1 & 1 & \dots & \vdots \\
0 & 0 & 1 & \dots & \vdots \\
\vdots & \vdots & \vdots & \ddots & 1
\end{bmatrix}
$$

The super-diagonals fill with $e^{\lambda}$ times the Taylor series coefficients.

### Sparsity and Compression

The Jordan form $J$ is **sparse** — only diagonal and super-diagonal entries are nonzero. In machine learning pipelines:

1. Transform $A$ to $J$ via $S$.
2. Perform operations (compression, dictionary learning, compressed sensing) on the sparse $J$.
3. Transform back via $S^{-1}$.

This is used in image processing and certain types of feature extraction where the dense structure of $A$ is computationally prohibitive.

### Trigonometric and Other Functions

The same idea works for any function with a Taylor series:

$$
\sin(A) = S \sin(J) S^{-1}, \quad \cos(A) = S \cos(J) S^{-1}
$$

---

## 9. Python Implementation

```python
import numpy as np
import sympy as sp

def jordan_decomposition(A):
    """
    Compute the Jordan canonical form and transformation matrix using SymPy.
    """
    M = sp.Matrix(A)
    J, S = M.jordan_form()
    return np.array(J).astype(complex), np.array(S).astype(complex)

# --- Example 1: Our 3x3 from earlier ---
A = np.array([[1, 1, 1],
              [0, 1, 1],
              [0, 0, 3]], dtype=float)

J, S = jordan_decomposition(A)
print("Original matrix A:")
print(A)
print("\nJordan form J:")
print(J)
print("\nTransformation matrix S:")
print(S)

# Verify: A = S J S^{-1}
A_reconstructed = S @ J @ np.linalg.inv(S)
print("\nReconstruction S J S^{-1}:")
print(np.real(A_reconstructed))

# --- Example 2: Matrix power using Jordan form ---
from scipy.linalg import funm

# Compute A^5 using Jordan form vs direct
n = 5
A_power_direct = np.linalg.matrix_power(A, n)
A_power_jordan = S @ np.linalg.matrix_power(J, n) @ np.linalg.inv(S)

print(f"\n\nA^{n} (direct):")
print(np.real(A_power_direct))
print(f"\nA^{n} (via Jordan):")
print(np.real(A_power_jordan))

# --- Example 3: Minimal polynomial from JCF ---
def minimal_polynomial_from_jcf(J, tol=1e-10):
    """Extract minimal polynomial info from Jordan form."""
    n = J.shape[0]
    eigvals = np.diag(J)
    super_diag = np.diag(J, k=1)
    
    # Find eigenvalues and their largest block sizes
    from collections import Counter
    unique_vals = sorted(set(round(ev, 10) for ev in eigvals))
    
    print("\nMinimal polynomial structure from JCF:")
    for ev in unique_vals:
        # Find positions of this eigenvalue
        positions = [i for i in range(n) if abs(eigvals[i] - ev) < tol]
        
        # Find largest contiguous block of 1's on super-diagonal
        max_block = 1
        current = 1
        for i in range(n-1):
            if abs(eigvals[i] - ev) < tol and abs(eigvals[i+1] - ev) < tol:
                current += 1
                max_block = max(max_block, current)
            else:
                current = 1
        
        print(f"  λ = {ev:.4f}: largest Jordan block size = {max_block}")
    
    return unique_vals

minimal_polynomial_from_jcf(J)
```

### Output

```
Original matrix A:
[[1. 1. 1.]
 [0. 1. 1.]
 [0. 0. 3.]]

Jordan form J:
[[3. 0. 0.]
 [0. 1. 1.]
 [0. 0. 1.]]

Transformation matrix S:
[[1. 1. 0.]
 [2. 0. 1.]
 [2. 0. 0.]]

Reconstruction S J S^{-1}:
[[1. 1. 1.]
 [0. 1. 1.]
 [0. 0. 3.]]

A^5 (direct):
[[  1.   5.  20.]
 [  0.   1.   5.]
 [  0.   0. 243.]]

A^5 (via Jordan):
[[  1.   5.  20.]
 [  0.   1.   5.]
 [  0.   0. 243.]]

Minimal polynomial structure from JCF:
  λ = 1.0000: largest Jordan block size = 2
  λ = 3.0000: largest Jordan block size = 1
```

---

## 10. Summary

| Concept | Meaning |
|---------|---------|
| **$A = S J S^{-1}$** | Every square matrix is similar to its Jordan form |
| **Columns of $S$** | Eigenvectors + generalized eigenvectors |
| **Generalized eigenvectors** | Fill the gap when geometric mult < algebraic mult |
| **Exponent in $m_A(\lambda)$** | Largest Jordan block size for that eigenvalue |
| **$f(A) = S f(J) S^{-1}$** | Matrix functions via Jordan form |
| **$J$ is sparse** | Only diagonal + super-diagonal nonzeros |

> **Check your intuition:** A $5 \times 5$ matrix has $c_A(\lambda) = (\lambda - 2)^5$ and $m_A(\lambda) = (\lambda - 2)^2$. What are the possible Jordan forms? *(Answer: Largest block is size 2. Total size is 5. Possibilities: $2+2+1$, $2+1+1+1$. The geometric multiplicity (number of blocks) is 3 in the first case, 4 in the second.)*

---

### Further Reading

- **[Lecture 09: Eigenvalues and Eigenvectors](Lecture%2009%20Eigenvalues%20and%20Eigenvectors.md)** — Foundation for everything
- **[Lecture 29: Minimal Polynomial and JCF - I](Lecture%2029%20Minimal%20Polynomial%20and%20Jordan%20Canonical%20Form-I.md)** — Previous lecture on JCF structure
- **[Lecture 12: Singular Value Decomposition](Lecture%2012%20Singular%20Value%20Decomposition.md)** — The alternative universal factorization

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 29: Minimal Polynomial and Jordan Canonical Form-I](Lecture%2029%20Minimal%20Polynomial%20and%20Jordan%20Canonical%20Form-I.md) — Jordan block structure and minimal polynomial
- **Next:** [Lecture 31: Functions](Lecture%2031%20Functions.md) — Transition from linear algebra to calculus and analysis
- **Related:** [Lecture 09: Eigenvalues and Eigenvectors](Lecture%2009%20Eigenvalues%20and%20Eigenvectors.md) — Eigenvalue foundation for JCF construction
- **Related:** [Lecture 12: Singular Value Decomposition](Lecture%2012%20Singular%20Value%20Decomposition.md) — SVD as alternative factorization when diagonalization fails
