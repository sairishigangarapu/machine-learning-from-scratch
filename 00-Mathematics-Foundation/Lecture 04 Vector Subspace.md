## Vector Subspaces

*Essential Mathematics for ML — Structured Notes*

---

## 1. Definition of a Subspace

Let $V$ be a vector space over $\mathbb{R}$. A subset $S \subseteq V$ is defined as a **subspace** if $S$ itself is a vector space under the same addition and scalar multiplication operations defined on $V$.

Geometrically, a subspace in $\mathbb{R}^n$ is an "origin-anchored flat slice" of the larger space. To be a valid subspace, it must be "flat" (linear) and pass through the origin.

---

## 2. The Three-Point Subspace Criterion

Rather than checking all vector space axioms, a subset $S$ is a subspace if and only if it satisfies these three conditions:

1. **Existence of Zero Vector:** $\mathbf{0} \in S$.
2. **Closure under Addition:** If $\mathbf{x}, \mathbf{y} \in S$, then $\mathbf{x} + \mathbf{y} \in S$.
3. **Closure under Scalar Multiplication:** If $\mathbf{x} \in S$ and $\alpha \in \mathbb{R}$, then $\alpha \mathbf{x} \in S$.

**Unified Theorem:** $S$ is a subspace if for all $a, b \in \mathbb{R}$ and $\mathbf{u}, \mathbf{v} \in S$, the linear combination $a\mathbf{u} + b\mathbf{v} \in S$.

---

## 3. Examples and Non-Examples

### Valid Subspaces

* **Trivial Subspaces:** For any vector space $V$, the set containing only the zero vector $\{\mathbf{0}\}$ and the space $V$ itself are subspaces.
* **Symmetric Matrices:** In the space of $3 \times 3$ matrices, the set of all symmetric matrices ($A = A^T$) is a subspace.
* **Homogeneous Planes:** A plane defined by $x_1 + x_2 - x_3 = 0$ is a subspace because it contains $(0,0,0)$ and is closed under linear combinations.

### Non-Examples

* **Affine Planes (Non-homogeneous):** The set $x_1 + x_2 + x_3 = 1$ is **not** a subspace. It fails the zero vector check ($0+0+0 \neq 1$) and closure under scaling.
* **Unions:** The union of two subspaces $S_1 \cup S_2$ is generally **not** a subspace unless $S_1 \subseteq S_2$ or $S_2 \subseteq S_1$. However, the **intersection** $S_1 \cap S_2$ is always a subspace.

---

## 4. Linear Span

The **span** of a set of vectors $S = \{\mathbf{v}_1, \dots, \mathbf{v}_n\}$ is the set of all possible linear combinations of those vectors:


$$\text{span}(S) = \{c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_n \mathbf{v}_n : c_i \in \mathbb{R}\}$$

* **Property:** $\text{span}(S)$ is the **smallest subspace** that contains all vectors in $S$.
* **Example:** In $\mathbb{R}^3$, the span of two non-collinear vectors is a plane passing through the origin.

---

## 5. Matrix-Related Subspaces

For any $m \times n$ matrix $A$, there are four fundamental subspaces that are critical for understanding data transformations:

### Row Space, $\text{Row}(A)$

The span of the row vectors of $A$. It is a subspace of $\mathbb{R}^n$.

### Column Space, $\text{Col}(A)$

The span of the column vectors of $A$. It is a subspace of $\mathbb{R}^m$. In ML, this is the **Range** of the matrix, representing all possible outputs of the transformation $A\mathbf{x}$.

### Null Space, $N(A)$

The set of all vectors $\mathbf{x}$ such that $A\mathbf{x} = \mathbf{0}$.


$$N(A) = \{ \mathbf{x} \in \mathbb{R}^n : A\mathbf{x} = \mathbf{0} \}$$


It is a subspace of $\mathbb{R}^n$ and represents the directions that are "collapsed" to zero by the matrix.

---

## 6. Significance in Machine Learning

Subspaces are the foundation for several core ML techniques:

* **Dimensionality Reduction (PCA):** Finds a lower-dimensional subspace that captures the maximum variance of the data.
* **SVD (Singular Value Decomposition):** Decomposes a matrix into components related to these four fundamental subspaces.
* **Linear Regression:** Seeks to project the target vector onto the column space of the feature matrix.

---
