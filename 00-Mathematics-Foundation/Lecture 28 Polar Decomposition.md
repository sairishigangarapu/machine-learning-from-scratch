## Polar Decomposition

*The Matrix Analogue of $z = r e^{i\theta}$ — Structured Notes*

---

## 0. Key Terminology — What All These Words Mean

Before we touch a single matrix, let's get every term crystal clear so nothing trips you up later.

### Orthogonal Matrix ($W$)

A square matrix $Q$ is **orthogonal** if $Q^T Q = Q Q^T = I$. This means:
- Columns are **orthonormal** — each pair of columns has dot product zero, and each column has length 1.
- $Q^{-1} = Q^T$ — the inverse is just the transpose. This makes computations stupidly cheap.
- Geometrically: an orthogonal matrix is a **pure rotation or reflection**. It preserves lengths ($\|Qx\| = \|x\|$) and angles ($\langle Qx, Qy \rangle = \langle x, y \rangle$). It can spin your data around but never stretch or squash it.

*Think of it as:* A rigid metal rod — you can rotate it any way you want, but it stays the same length.

### Positive Semi-Definite Matrix ($P$)

A symmetric matrix $P$ is **positive semi-definite (PSD)** if:
- $P^T = P$ (it's symmetric).
- $x^T P x \ge 0$ for every vector $x$.
- All eigenvalues of $P$ are $\ge 0$. If all eigenvalues are $> 0$, it's **positive definite (PD)**.

*Think of it as:* A stretchy rubber band — it can only pull or squash along its axes, never rotate.

### Unitary Matrix

The complex-number analogue of an orthogonal matrix. If a matrix has complex entries, $U$ is **unitary** if $U^* U = U U^* = I$ where $U^*$ is the conjugate transpose. All the same properties — preserves lengths and angles, just in complex space.

### Isometry

A transformation $T$ is an **isometry** if it preserves distances: $\|T(x) - T(y)\| = \|x - y\|$ for all $x, y$. Orthogonal and unitary matrices are examples of isometries. In polar decomposition, $W$ is an isometry — it rotates/reflects but doesn't change the shape.

### Orthonormal Columns (Rectangular Case)

For a rectangular $m \times n$ matrix $U$ (where $m \ge n$), **orthonormal columns** means $U^T U = I_n$. The columns are unit vectors that are pairwise perpendicular. But $U U^T \neq I_m$ (unless $m = n$) — the rows are NOT orthonormal.

*Think of it as:* A bundle of $n$ perfectly perpendicular sticks floating in $m$-dimensional space. There are fewer sticks than dimensions, so they only span an $n$-dimensional subspace.

### Singular Value Decomposition (SVD)

Every matrix $A$ (square or rectangular) can be factored as:

$$
A = U \Sigma V^T
$$

- $U$, $V$ are orthogonal matrices.
- $\Sigma$ is diagonal with **singular values** $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r \ge 0$ on the diagonal.

SVD is the Swiss Army knife of linear algebra — polar decomposition, PCA, pseudo-inverse, and low-rank approximation all come from it.

### Matrix Square Root ($\sqrt{A}$)

For a PSD matrix $A$, there exists a unique PSD matrix $B$ such that $B^2 = A$. We write $B = \sqrt{A}$. If $A = S \Lambda S^T$ (eigendecomposition), then $\sqrt{A} = S \sqrt{\Lambda} S^T$ where $\sqrt{\Lambda}$ has $\sqrt{\lambda_i}$ on the diagonal.

---

## 1. Motivation and Intuition

### The Complex Number Analogy

Every complex number $z = x + iy$ can be written in polar form:

$$
z = r e^{i\theta}
$$

where $r = \sqrt{x^2 + y^2} \ge 0$ is the magnitude and $e^{i\theta} = \cos\theta + i\sin\theta$ is a unit complex number (a rotation on the unit circle). You're decomposing the complex number into a **non-negative scaling factor** and a **pure rotation**.

Polar decomposition does the exact same thing for matrices. It factorizes any matrix $A$ into:

$$
A = WP
$$

where $W$ is an **orthogonal matrix** (a rotation/reflection — the analogue of $e^{i\theta}$) and $P$ is a **positive semi-definite matrix** (the analogue of $r \ge 0$).

### Why This Matters

In machine learning and computer graphics, you often want to separate the "stretching" part of a transformation from the "rotating" part. Polar decomposition gives you exactly that — a clean split between shape (stretch/squash encoded in $P$) and orientation (rotation/reflection encoded in $W$).

---

## 2. Formal Definition: Square Matrices

Let $A \in \mathbb{R}^{n \times n}$ be a square matrix. There exists an **orthogonal matrix** $W$ and a **positive semi-definite matrix** $P$ such that:

$$
\boxed{A = WP}
$$

- $W$ is orthogonal: $W^T W = W W^T = I$ (its columns are orthonormal).
- $P$ is positive semi-definite: $P^T = P$ and $x^T P x \ge 0$ for all $x$.

If $A$ is **invertible**, this decomposition is **unique**.

---

## 3. Connection to Singular Value Decomposition

The cleanest way to understand polar decomposition is through SVD. Every square matrix $A$ has an SVD:

$$
A = U \Sigma V^T
$$

where $U$ and $V$ are orthogonal and $\Sigma$ is diagonal with singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_n \ge 0$.

Now, insert $V^T V = I$ between $U$ and $\Sigma$:

$$
A = U (V^T V) \Sigma V^T = (U V^T) (V \Sigma V^T)
$$

Look at what we've got:

- **$W = U V^T$**: Product of two orthogonal matrices. Still orthogonal.
- **$P = V \Sigma V^T$**: A symmetric matrix ($P^T = P$) with eigenvalues $\sigma_i \ge 0$. This is precisely the definition of a positive semi-definite matrix — it's the orthogonal diagonalization of $P$.

So polar decomposition falls out of SVD for free.

---

## 4. Proof of Uniqueness (Invertible Case)

Suppose $A$ is invertible and admits two polar decompositions:

$$
A = W_1 P_1 = W_2 P_2
$$

Multiply both sides on the left by $W_2^T$:

$$
W_2^T W_1 P_1 = P_2
$$

Since $W_2^T W_1$ is orthogonal (product of orthogonal matrices), we have $P_2 = Q P_1$ where $Q$ is orthogonal.

Now $P_2^T P_2 = P_1 Q^T Q P_1 = P_1^2$. But $P_2^2 = P_2^T P_2 = P_1^2$ as well. Taking positive semi-definite square roots (which are unique for PSD matrices), we get:

$$
P_1 = P_2
$$

And then $W_1 = W_2$ follows immediately. The decomposition is unique.

---

## 5. Worked Example: Square Matrix

### Problem

Find the polar decomposition of:

$$
A = \begin{bmatrix} 11 & -5 \\ -2 & 10 \end{bmatrix}
$$

### Solution via SVD

**Step 1:** Compute $A^T A$.

$$
A^T A = \begin{bmatrix} 125 & -75 \\ -75 & 125 \end{bmatrix}
$$

Eigenvalues of $A^T A$ are $\lambda = 200$ and $\lambda = 50$.

**Step 2:** Singular values.

$$
\sigma_1 = \sqrt{200} = 10\sqrt{2}, \quad \sigma_2 = \sqrt{50} = 5\sqrt{2}
$$

**Step 3:** Right singular vectors (eigenvectors of $A^T A$).

$$
v_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix}, \quad
v_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

**Step 4:** Left singular vectors.

$$
u_1 = \frac{A v_1}{\sigma_1} = \frac{1}{5} \begin{bmatrix} 4 \\ -3 \end{bmatrix}, \quad
u_2 = \frac{A v_2}{\sigma_2} = \frac{1}{5} \begin{bmatrix} 3 \\ 4 \end{bmatrix}
$$

**Step 5:** Assemble SVD.

$$
A = U \Sigma V^T = \frac{1}{5} \begin{bmatrix} 4 & 3 \\ -3 & 4 \end{bmatrix}
\begin{bmatrix} 10\sqrt{2} & 0 \\ 0 & 5\sqrt{2} \end{bmatrix}
\frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix}
$$

**Step 6:** Extract $W$ and $P$.

$$
W = U V^T = \frac{1}{5\sqrt{2}} \begin{bmatrix} 7 & -1 \\ 1 & 7 \end{bmatrix}
$$

$$
P = V \Sigma V^T = \frac{5}{\sqrt{2}} \begin{bmatrix} 3 & -1 \\ -1 & 3 \end{bmatrix}
$$

**Verification:**

$$
W P = \frac{1}{5\sqrt{2}} \begin{bmatrix} 7 & -1 \\ 1 & 7 \end{bmatrix}
\cdot \frac{5}{\sqrt{2}} \begin{bmatrix} 3 & -1 \\ -1 & 3 \end{bmatrix}
= \begin{bmatrix} 11 & -5 \\ -2 & 10 \end{bmatrix} = A
$$

$W$ is orthogonal ($W^T W = I$) and $P$ is symmetric with eigenvalues $10\sqrt{2}, 5\sqrt{2} \ge 0$.

---

## 6. Polar Decomposition for Rectangular Matrices

For rectangular matrices, there are two variants depending on the shape.

### Right Polar Decomposition ($m \ge n$)

If $A$ is $m \times n$ with $m \ge n$ (a "tall" matrix), the **right polar decomposition** is:

$$
\boxed{A = U P}
$$

- $U$ is $m \times n$ with orthonormal columns ($U^T U = I_n$).
- $P$ is $n \times n$ positive semi-definite.

The "right" refers to $P$ sitting on the **right** side of the decomposition.

### Left Polar Decomposition ($m \le n$)

If $A$ is $m \times n$ with $m \le n$ (a "fat" matrix), the **left polar decomposition** is:

$$
\boxed{A = H U}
$$

- $H$ is $m \times m$ positive semi-definite.
- $U$ is $m \times n$ with orthonormal rows ($U U^T = I_m$).

The "left" refers to $H$ sitting on the **left** side.

---

## 7. Worked Example: Tall Matrix

### Problem

Find the polar decomposition of:

$$
A = \begin{bmatrix} 3 & 1 \\ 0 & 1 \\ 1 & 0 \end{bmatrix}
$$

This is a $3 \times 2$ matrix ($m = 3 > n = 2$), so we use the **right polar decomposition** $A = U P$.

### Solution

**Step 1:** Compute $A^T A$.

$$
A^T A = \begin{bmatrix} 10 & 3 \\ 3 & 2 \end{bmatrix}
$$

**Step 2:** Eigendecomposition of $A^T A$.

$$
A^T A = S B S^T
$$

where

$$
S = \begin{bmatrix}
\frac{3}{\sqrt{10}} & \frac{1}{\sqrt{10}} \\[4pt]
\frac{1}{\sqrt{10}} & -\frac{3}{\sqrt{10}}
\end{bmatrix}, \quad
B = \begin{bmatrix} 11 & 0 \\ 0 & 1 \end{bmatrix}
$$

**Step 3:** Compute $P = \sqrt{A^T A}$.

Since $A^T A = S B S^T$, the square root is:

$$
P = S B^{1/2} S^T
$$

where $B^{1/2} = \begin{bmatrix} \sqrt{11} & 0 \\ 0 & 1 \end{bmatrix}$.

$$
P = \frac{1}{10} \begin{bmatrix}
9\sqrt{11} + 1 & 3\sqrt{11} - 3 \\[4pt]
3\sqrt{11} - 3 & \sqrt{11} + 9
\end{bmatrix}
$$

**Step 4:** Compute $U = A P^{-1}$.

$$
U = A P^{-1}
$$

The resulting $U$ is $3 \times 2$ with orthonormal columns, and $A = U P$ is the right polar decomposition.

---

## 8. Worked Example: Fat Matrix

### Problem

Find the polar decomposition of:

$$
A = \begin{bmatrix} 1 & 0 & 1 \\ 1 & 1 & 0 \end{bmatrix}
$$

This is a $2 \times 3$ matrix ($m = 2 < n = 3$), so we use the **left polar decomposition** $A = H U$.

### Solution

**Step 1:** Compute $A A^T$.

$$
A A^T = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}
$$

**Step 2:** Eigendecomposition of $A A^T$.

$$
A A^T = S B S^T
$$

where

$$
S = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \quad
B = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}
$$

**Step 3:** Compute $H = \sqrt{A A^T}$.

$$
H = S B^{1/2} S^T = \frac{1}{2} \begin{bmatrix}
\sqrt{3} + 1 & \sqrt{3} - 1 \\[4pt]
\sqrt{3} - 1 & \sqrt{3} + 1
\end{bmatrix}
$$

**Step 4:** Compute $U = H^{-1} A$.

The resulting $U$ is $2 \times 3$ with orthonormal rows, and $A = H U$ is the left polar decomposition.

---

## 9. Python Implementation

```python
import numpy as np

def polar_decomposition(A):
    """
    Compute the polar decomposition A = W P for a square matrix A.
    Uses SVD under the hood.
    """
    U, S, Vt = np.linalg.svd(A)
    W = U @ Vt
    P = Vt.T @ np.diag(S) @ Vt
    return W, P

# --- Square matrix example ---
A_square = np.array([[11, -5], [-2, 10]], dtype=float)
W, P = polar_decomposition(A_square)

print("Square matrix A:")
print(A_square)
print("\nW (orthogonal):")
print(W)
print("\nP (positive semi-definite):")
print(P)
print("\nReconstruction W @ P:")
print(W @ P)
print("\nW^T W (should be identity):")
print(W.T @ W)

# --- Tall matrix (right polar decomposition) ---
A_tall = np.array([[3, 1], [0, 1], [1, 0]], dtype=float)

# Right polar: A = U P where U has orthonormal columns
U, S, Vt = np.linalg.svd(A_tall, full_matrices=False)
U_right = U @ Vt          # m x n with orthonormal columns
P_right = Vt.T @ np.diag(S) @ Vt  # n x n PSD

print("\n\nTall matrix A (3x2):")
print(A_tall)
print("\nU (orthonormal columns):")
print(U_right)
print("\nP (PSD):")
print(P_right)
print("\nReconstruction U @ P:")
print(U_right @ P_right)
print("\nU^T U (should be identity):")
print(U_right.T @ U_right)
```

### Output

```
Square matrix A:
[[11. -5.]
 [-2. 10.]]

W (orthogonal):
[[ 0.98994949 -0.14142136]
 [ 0.14142136  0.98994949]]

P (positive semi-definite):
[[10.60660172 -3.53553391]
 [-3.53553391 10.60660172]]

Reconstruction W @ P:
[[11. -5.]
 [-2. 10.]]

W^T W (should be identity):
[[ 1.00000000e+00 -6.12323400e-17]
 [-6.12323400e-17  1.00000000e+00]]
```

---

## 10. Properties at a Glance

| Property | Detail |
|----------|--------|
| **Existence** | Every matrix (square or rectangular) has a polar decomposition |
| **Uniqueness** | Unique if $A$ is invertible (square) or full rank (rectangular) |
| **SVD shortcut** | $W = UV^T$, $P = V \Sigma V^T$ |
| **$P$ is always** | $P = \sqrt{A^T A}$ (the unique PSD square root) |
| **$W$ is always** | The "rotation" that aligns the stretch axes |
| **Polar vs. SVD** | SVD gives three factors ($U\Sigma V^T$); Polar gives two ($WP$) but $W$ loses the separate left/right rotation information |

> **Check your intuition:** If $A$ is an orthogonal matrix itself, what does its polar decomposition look like? *(Answer: $A = W P$ with $W = A$ and $P = I$. An orthogonal matrix is already a pure rotation with no stretching.)*

---

## 11. Applications in Machine Learning

- **Computer graphics & robotics:** Polar decomposition separates the rigid-body rotation ($W$) from the stretching/squashing ($P$) of a transformation, critical for physics simulations and animation.
- **Procrustes analysis:** Finding the optimal orthogonal transformation to align two point clouds uses $W = UV^T$ from the SVD — which is exactly the polar factor.
- **Numerical optimization:** Some optimization algorithms in manifold learning use polar decomposition to project matrices back onto the orthogonal group.
- **Neural network initialization:** Orthogonal initializations (to prevent vanishing/exploding gradients) can be generated via polar decomposition of random matrices.

---

### Further Reading

- **[Lecture 12: Singular Value Decomposition](Lecture%2012%20Singular%20Value%20Decomposition.md)** — SVD is the engine behind polar decomposition
- **[Lecture 13: SVD — Properties and Applications](Lecture%2013%20SVD%20%20Properties%20and%20Applications.md)** — More on how SVD factorizations are used
- **[Lecture 14: Low Rank Approximations](Lecture%2014%20Low%20Rank%20Approximations.md)** — Truncating SVD for compression
