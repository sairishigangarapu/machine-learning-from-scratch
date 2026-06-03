## Gram-Schmidt Process

*Turning Any Basis into an Orthogonal (or Orthonormal) One — Structured Notes*

---

## 0. Key Terminology — What All These Words Mean

### Vector Space ($V$)

A set of vectors that is closed under **addition** and **scalar multiplication**. If you add any two vectors in $V$, you get another vector in $V$. If you scale any vector by a number, you stay in $V$.

*Think of it as*: The entire universe of possible vectors you're allowed to work with.

### Inner Product Space

A vector space equipped with an **inner product** $\langle x, y \rangle$ — a way to measure angles and lengths. The dot product in $\mathbb{R}^n$ is the classic example. An inner product lets you define orthogonality ($\langle x, y \rangle = 0$) and norms ($\|x\| = \sqrt{\langle x, x \rangle}$).

### Linear Independence

A set of vectors is **linearly independent** if no vector can be written as a combination of the others. Formally: $\alpha_1 v_1 + \alpha_2 v_2 + \dots + \alpha_k v_k = 0$ only when all $\alpha_i = 0$.

*Think of it as*: Each vector brings something new to the table. None of them is redundant.

### Linear Dependence

If at least one vector IS a combination of the others. In Gram-Schmidt, if you feed in linearly dependent vectors, the process will produce a zero vector and fail at normalization.

### Basis

A set of vectors that is (1) linearly independent AND (2) spans the entire space. Every vector in the space can be written as a **unique** combination of the basis vectors. The number of vectors in a basis is the **dimension** of the space.

### Span

The set of ALL possible vectors you can reach by taking linear combinations of a given set of vectors.

$$
\text{span}\{v_1, \dots, v_k\} = \{\alpha_1 v_1 + \dots + \alpha_k v_k : \alpha_i \in \mathbb{R}\}
$$

### Subspace ($V_0$)

A subset of a vector space that is itself a vector space (closed under addition and scalar multiplication). Think of it as a flat plane or line passing through the origin inside a larger space.

### Orthogonal Set

A set of vectors where every pair is perpendicular: $\langle v_i, v_j \rangle = 0$ for $i \neq j$. Key property: orthogonal sets are automatically linearly independent.

### Orthonormal Set

An orthogonal set where every vector also has **unit length**: $\|v_i\| = 1$ for all $i$. These are the nicest possible vectors to work with because they make projections trivial.

### Orthogonal Projection

The "shadow" of a vector $x$ onto a subspace $V_0$. Given an orthogonal basis $\{v_1, \dots, v_k\}$ of $V_0$, the projection is:

$$
p = \frac{\langle x, v_1 \rangle}{\langle v_1, v_1 \rangle} v_1 + \dots + \frac{\langle x, v_k \rangle}{\langle v_k, v_k \rangle} v_k
$$

This is the **closest point** in $V_0$ to $x$.

### Orthogonal Complement ($V_0^\perp$)

The set of ALL vectors that are orthogonal to **every** vector in a subspace $V_0$. Every vector can be uniquely split into a part in $V_0$ and a part in $V_0^\perp$.

### Gram-Schmidt Process

An algorithm that takes a linearly independent set $\{x_1, \dots, x_n\}$ and produces an orthogonal (or orthonormal) set $\{v_1, \dots, v_n\}$ spanning the same space. It works by iteratively subtracting the projection of each new vector onto the already-constructed ones.

### QR Decomposition

A matrix factorization $A = QR$ where $Q$ has orthonormal columns and $R$ is upper triangular. QR decomposition *is* Gram-Schmidt under the hood — it's just the matrix form of the same process. In numerical linear algebra, `np.linalg.qr()` is the standard way to compute Gram-Schmidt because it uses numerically stable Householder reflections.

### Numerical Stability

How resistant an algorithm is to floating-point rounding errors. **Classical Gram-Schmidt** is numerically unstable — when vectors are nearly parallel, cancellation errors pile up. **Modified Gram-Schmidt** and **Householder QR** fix this by processing projections differently.

---

## 1. Motivation and Intuition

### Why Do We Need This?

In machine learning, orthonormal bases are everywhere. PCA gives you principal components that are orthonormal. SVD gives you left and right singular vectors that are orthonormal. Orthonormal matrices preserve distances, don't amplify gradients, and make projections trivial to compute.

But here's the problem: you don't always start with an orthonormal basis. You might have a set of linearly independent vectors that are messy — pointing in random directions with arbitrary lengths. The **Gram-Schmidt process** is a systematic procedure that takes any linearly independent set and transforms it into an orthogonal (or orthonormal) set that spans the exact same subspace.

Think of it like straightening out a pile of bent wire hangers — you're not changing what they're made of, you're just reshaping them so they all hang perfectly perpendicular to each other.

---

## 2. Orthogonal and Orthonormal Sets (Recap)

Before we build orthogonal vectors, let's recall what they are.

Let $V$ be an inner product space. Nonzero vectors $\{v_1, v_2, \dots, v_k\}$ form an **orthogonal set** if:

$$
\langle v_i, v_j \rangle = 0 \quad \text{whenever } i \neq j
$$

If, in addition, each vector has unit norm ($\|v_i\| = 1$ for all $i$), they form an **orthonormal set**.

### Key Fact

If a set of vectors is orthogonal, it is automatically **linearly independent**. The converse is not true — linear independence does not imply orthogonality. That's exactly why Gram-Schmidt exists.

---

## 3. Orthogonal Projection onto a Subspace (Recap)

Let $V$ be an inner product space and $V_0$ a finite-dimensional subspace of $V$. Then **any** vector $x \in V$ can be uniquely decomposed as:

$$
x = p + o
$$

where:
- $p \in V_0$ is the orthogonal projection of $x$ onto $V_0$
- $o \in V_0^\perp$ is perpendicular to every vector in $V_0$

The distance from $x$ to the subspace $V_0$ is simply $\|o\|$ — the length of the perpendicular component.

If $\{v_1, v_2, \dots, v_n\}$ is an **orthogonal basis** for $V_0$, the projection $p$ is easy to compute:

$$
p = \frac{\langle x, v_1 \rangle}{\langle v_1, v_1 \rangle} v_1 + \frac{\langle x, v_2 \rangle}{\langle v_2, v_2 \rangle} v_2 + \dots + \frac{\langle x, v_n \rangle}{\langle v_n, v_n \rangle} v_n
$$

This formula is the engine that drives Gram-Schmidt.

---

## 4. The Gram-Schmidt Process

### Setup

Let $\{x_1, x_2, \dots, x_n\}$ be a basis of an inner product space $V$ (so they're linearly independent and span $V$). We want to construct an **orthogonal basis** $\{v_1, v_2, \dots, v_n\}$ that spans the same space.

### Step-by-Step Construction

**Step 1:** Set the first vector as-is.

$$
v_1 = x_1
$$

**Step 2:** For the second vector, take $x_2$ and **subtract its projection** onto $v_1$. This removes any component of $x_2$ that points in the $v_1$ direction, leaving only the part that's orthogonal to $v_1$.

$$
v_2 = x_2 - \frac{\langle x_2, v_1 \rangle}{\langle v_1, v_1 \rangle} v_1
$$

**Step 3:** For the third vector, subtract the projections onto both $v_1$ and $v_2$.

$$
v_3 = x_3 - \frac{\langle x_3, v_1 \rangle}{\langle v_1, v_1 \rangle} v_1 - \frac{\langle x_3, v_2 \rangle}{\langle v_2, v_2 \rangle} v_2
$$

**General Step:** For the $k$-th vector, subtract the projection onto all previously constructed $v_i$.

$$
v_k = x_k - \sum_{i=1}^{k-1} \frac{\langle x_k, v_i \rangle}{\langle v_i, v_i \rangle} v_i
$$

At the end, $\{v_1, v_2, \dots, v_n\}$ is an **orthogonal set** spanning the same space as $\{x_1, x_2, \dots, x_n\}$.

### Why Does This Work?

When we subtract the projection of $x_k$ onto the span of $\{v_1, \dots, v_{k-1}\}$, we're literally peeling away every component that lies in the subspace we've already built. What remains ($v_k$) is forced to be orthogonal to all previous $v_i$.

You can verify this directly:

$$
\langle v_1, v_2 \rangle = \left\langle v_1, x_2 - \frac{\langle x_2, v_1 \rangle}{\langle v_1, v_1 \rangle} v_1 \right\rangle = \langle v_1, x_2 \rangle - \frac{\langle x_2, v_1 \rangle}{\langle v_1, v_1 \rangle} \langle v_1, v_1 \rangle = 0
$$

---

## 5. Properties of Gram-Schmidt

1. **Span preservation:** $\text{span}\{v_1, v_2, \dots, v_k\} = \text{span}\{x_1, x_2, \dots, x_k\}$ for every $k$.

2. **Orthogonality to earlier vectors:** $v_k$ is orthogonal to $x_1, x_2, \dots, x_{k-1}$ (not just to $v_1, \dots, v_{k-1}$).

3. **Projection interpretation:** $v_k = x_k - p_k$, where $p_k$ is the orthogonal projection of $x_k$ onto $\text{span}\{x_1, \dots, x_{k-1}\}$.

4. **Distance interpretation:** $\|v_k\|$ is the distance from $x_k$ to the subspace spanned by $\{x_1, \dots, x_{k-1}\}$.

---

## 6. Making It Orthonormal

If you want an **orthonormal** basis instead of just orthogonal, normalize each vector after construction:

$$
w_1 = \frac{v_1}{\|v_1\|}, \quad
w_2 = \frac{v_2}{\|v_2\|}, \quad
\dots, \quad
w_n = \frac{v_n}{\|v_n\|}
$$

The process then becomes:

1. $v_1 = x_1$, $\quad w_1 = v_1 / \|v_1\|$
2. $v_2 = x_2 - \langle x_2, w_1 \rangle w_1$, $\quad w_2 = v_2 / \|v_2\|$
3. $v_3 = x_3 - \langle x_3, w_1 \rangle w_1 - \langle x_3, w_2 \rangle w_2$, $\quad w_3 = v_3 / \|v_3\|$

Working with unit vectors $w_i$ from the start keeps the formulas clean because $\langle v_i, v_i \rangle = 1$ for each $w_i$.

---

## 7. Worked Example

### Problem

Let $P$ be a plane in $\mathbb{R}^3$ spanned by:

$$
x_1 = \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}, \quad
x_2 = \begin{bmatrix} -1 \\ 0 \\ 2 \end{bmatrix}
$$

**Part (a):** Find an orthonormal basis for $P$.
**Part (b):** Extend it to an orthonormal basis for $\mathbb{R}^3$.

### Solution

**Step 1:** Set $v_1 = x_1$ and normalize.

$$
v_1 = \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}, \quad
\|v_1\| = \sqrt{1 + 4 + 4} = 3
$$

$$
w_1 = \frac{v_1}{\|v_1\|} = \frac{1}{3} \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}
$$

**Step 2:** Subtract the projection of $x_2$ onto $w_1$.

$$
\langle x_2, w_1 \rangle = (-1)(1/3) + (0)(2/3) + (2)(2/3) = -\frac{1}{3} + \frac{4}{3} = 1
$$

$$
v_2 = x_2 - \langle x_2, w_1 \rangle w_1
= \begin{bmatrix} -1 \\ 0 \\ 2 \end{bmatrix} - 1 \cdot \frac{1}{3} \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}
= \begin{bmatrix} -1 - \frac{1}{3} \\ 0 - \frac{2}{3} \\ 2 - \frac{2}{3} \end{bmatrix}
= \begin{bmatrix} -\frac{4}{3} \\ -\frac{2}{3} \\ \frac{4}{3} \end{bmatrix}
$$

Now normalize $v_2$:

$$
\|v_2\| = \sqrt{\frac{16}{9} + \frac{4}{9} + \frac{16}{9}} = \sqrt{\frac{36}{9}} = \sqrt{4} = 2
$$

$$
w_2 = \frac{v_2}{\|v_2\|} = \frac{1}{2} \begin{bmatrix} -\frac{4}{3} \\ -\frac{2}{3} \\ \frac{4}{3} \end{bmatrix}
= \begin{bmatrix} -\frac{2}{3} \\ -\frac{1}{3} \\ \frac{2}{3} \end{bmatrix}
$$

So $\{w_1, w_2\}$ is an orthonormal basis for $P$.

**Part (b):** Extend to $\mathbb{R}^3$.

We need a third vector $x_3$ that is linearly independent of $x_1$ and $x_2$. The simplest choice is $x_3 = (0, 0, 1)$. Apply the same process:

$$
\langle x_3, w_1 \rangle = (0)(1/3) + (0)(2/3) + (1)(2/3) = \frac{2}{3}
$$

$$
\langle x_3, w_2 \rangle = (0)(-2/3) + (0)(-1/3) + (1)(2/3) = \frac{2}{3}
$$

$$
v_3 = x_3 - \langle x_3, w_1 \rangle w_1 - \langle x_3, w_2 \rangle w_2
= \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
- \frac{2}{3} \cdot \frac{1}{3} \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}
- \frac{2}{3} \cdot \frac{1}{3} \begin{bmatrix} -2 \\ -1 \\ 2 \end{bmatrix}
$$

$$
v_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
- \begin{bmatrix} 2/9 \\ 4/9 \\ 4/9 \end{bmatrix}
- \begin{bmatrix} -4/9 \\ -2/9 \\ 4/9 \end{bmatrix}
= \begin{bmatrix} 2/9 \\ -2/9 \\ 1/9 \end{bmatrix}
= \frac{1}{9} \begin{bmatrix} 2 \\ -2 \\ 1 \end{bmatrix}
$$

Normalize:

$$
\|v_3\| = \frac{\sqrt{4 + 4 + 1}}{9} = \frac{3}{9} = \frac{1}{3}
$$

$$
w_3 = \frac{v_3}{\|v_3\|} = \frac{1}{3} \begin{bmatrix} 2 \\ -2 \\ 1 \end{bmatrix}
$$

Now $\{w_1, w_2, w_3\}$ is an orthonormal basis for $\mathbb{R}^3$.

---

## 8. Python Implementation

```python
import numpy as np

def gram_schmidt(X):
    """
    Given a matrix X whose columns are linearly independent vectors,
    returns an orthonormal matrix Q (same shape) with orthonormal columns.
    
    This is the "modified" Gram-Schmidt for numerical stability.
    """
    n, m = X.shape
    Q = np.zeros((n, m))
    
    for j in range(m):
        v = X[:, j].copy()
        for i in range(j):
            # Subtract projection onto previously computed q_i
            q_i = Q[:, i]
            v = v - np.dot(q_i, X[:, j]) * q_i
        
        # Normalize
        Q[:, j] = v / np.linalg.norm(v)
    
    return Q

# --- Example: Our plane in R^3 ---
X = np.array([
    [1, -1],
    [2,  0],
    [2,  2]
], dtype=float)

Q = gram_schmidt(X)
print("Orthonormal basis for the plane P:")
print(Q)

# Verify orthonormality
print("\nQ^T Q (should be identity):")
print(Q.T @ Q)

# --- Extend to R^3 ---
X_ext = np.array([
    [1, -1, 0],
    [2,  0, 0],
    [2,  2, 1]
], dtype=float)

Q_ext = gram_schmidt(X_ext)
print("\nOrthonormal basis for R^3:")
print(Q_ext)

print("\nQ^T Q (should be identity):")
print(Q_ext.T @ Q_ext)
```

### Output

```
Orthonormal basis for the plane P:
[[ 0.33333333 -0.66666667]
 [ 0.66666667 -0.33333333]
 [ 0.66666667  0.66666667]]

Q^T Q (should be identity):
[[ 1.00000000e+00 -5.55111512e-17]
 [-5.55111512e-17  1.00000000e+00]]

Orthonormal basis for R^3:
[[ 0.33333333 -0.66666667  0.66666667]
 [ 0.66666667 -0.33333333 -0.66666667]
 [ 0.66666667  0.66666667  0.33333333]]

Q^T Q (should be identity):
[[ 1.00000000e+00  0.00000000e+00  0.00000000e+00]
 [ 0.00000000e+00  1.00000000e+00  0.00000000e+00]
 [ 0.00000000e+00  0.00000000e+00  1.00000000e+00]]
```

### A Note on Numerical Stability

The classical Gram-Schmidt we described above can suffer from catastrophic cancellation in floating point arithmetic when vectors are nearly parallel. In practice, use the **modified Gram-Schmidt** (which processes each projection one at a time using the updated vector) or just call `np.linalg.qr(X, mode='reduced')` which does the same thing under the hood with Householder reflections — much more numerically stable.

```python
# The "cheat code" — QR decomposition gives you Gram-Schmidt for free
Q, R = np.linalg.qr(X_ext)
print("QR decomposition gives the same orthonormal basis:")
print(Q)
```

---

## 9. Connection to QR Decomposition

The Gram-Schmidt process is intimately related to **QR decomposition**. If $A$ is an $m \times n$ matrix with linearly independent columns, then:

$$
A = QR
$$

where $Q$ has orthonormal columns (the Gram-Schmidt output) and $R$ is an upper-triangular matrix containing the projection coefficients.

$$
R_{ij} = \begin{cases}
\|v_i\| & i = j \\
\langle x_j, w_i \rangle & i < j \\
0 & i > j
\end{cases}
$$

This is why `np.linalg.qr` is the standard numerical implementation of Gram-Schmidt.

> **Check your intuition:** What happens if you try to apply Gram-Schmidt to a set of vectors that are *not* linearly independent? *(Answer: At some step $k$, the vector $x_k$ will be entirely in the span of the previous vectors, so $v_k = 0$ and you can't normalize. Gram-Schmidt will detect linear dependence by producing a zero vector.)*

---

## 10. Why This Matters in Machine Learning

- **PCA** finds principal components by eigendecomposition of the covariance matrix. The eigenvectors are orthonormal — Gram-Schmidt guarantees we can always construct such bases.
- **Feature space orthogonalization** in kernel methods removes redundancy between basis functions.
- **Numerical optimization** algorithms (conjugate gradient, BFGS) use Gram-Schmidt-like ideas to maintain conjugate search directions.
- **QR decomposition** via Gram-Schmidt is the workhorse for solving least-squares problems in linear regression ($A x = b$ when $A$ is tall).

---

### Further Reading

- **[Lecture 08: Orthogonal Complement and Projection Mapping](Lecture%2008%20Orthogonal%20Complement%20and%20Projection%20Mapping.md)** — The projection formula that Gram-Schmidt relies on
- **[Lecture 12: Singular Value Decomposition](Lecture%2012%20Singular%20Value%20Decomposition.md)** — SVD gives you orthonormal bases for any matrix
- **[Lecture 16: Principal Component Analysis](Lecture%2016%20Principal%20Component%20Analysis.md)** — PCA finds orthonormal directions of maximum variance
