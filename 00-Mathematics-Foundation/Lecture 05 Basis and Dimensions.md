## Basis and Dimension

*Essential Mathematics for ML — Structured Notes*

---

## 1. Definition of a Basis

In Machine Learning, we often decompose complex data into a weighted sum of simpler components (e.g., PCA or Wavelet transforms). A **basis** is the set of vectors that allows for this unique representation.

A set $B = \{v_1, v_2, \dots, v_n\} \subseteq V$ is a basis of a vector space $V$ if it satisfies two conditions:

1. **Linear Independence:** No vector in $B$ can be expressed as a linear combination of the other vectors in $B$.
2. **Spanning Property:** Every vector $\mathbf{v} \in V$ can be written as a linear combination of the vectors in $B$:

$$\mathbf{v} = \alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \dots + \alpha_n \mathbf{v}_n$$



---

## 2. Finite vs. Infinite Dimensional Spaces

The "size" of a vector space is determined by its basis:

* **Finite-dimensional:** The basis contains a finite number of vectors.
* *Example:* $\mathbb{R}^n$ has dimension $n$.


* **Infinite-dimensional:** The basis requires an infinite set of vectors.
* *Example:* The space of all continuous functions or the space of all polynomials.



---

## 3. Standard and Custom Bases

### Standard Bases

These are the most intuitive bases used in computation:

* **In $\mathbb{R}^2$:** $e_1 = (1,0), e_2 = (0,1)$.
* **In $\mathbb{R}^n$:** $e_1 = (1,0,\dots,0), \dots, e_n = (0,0,\dots,1)$.
* **In Matrices ($M_{2 \times 2}$):**

$$\begin{bmatrix}1 & 0 \\ 0 & 0\end{bmatrix}, \begin{bmatrix}0 & 1 \\ 0 & 0\end{bmatrix}, \begin{bmatrix}0 & 0 \\ 1 & 0\end{bmatrix}, \begin{bmatrix}0 & 0 \\ 0 & 1\end{bmatrix}$$



The dimension here is $mn$ (for a $2 \times 2$, $\dim = 4$).

### Symmetric Matrix Basis

For a $2 \times 2$ symmetric matrix $\begin{bmatrix} a & b \\ b & c \end{bmatrix}$, the basis is:


$$\begin{bmatrix}1&0\\0&0\end{bmatrix}, \begin{bmatrix}0&1\\1&0\end{bmatrix}, \begin{bmatrix}0&0\\0&1\end{bmatrix}$$


The dimension is 3, reflecting the 3 degrees of freedom ($a, b, c$).

---

## 4. Fundamental Theorems

* **Theorem 1:** In an $n$-dimensional space, any set with more than $n$ vectors is **linearly dependent**.
* **Theorem 2:** Any linearly independent set can be **extended** to form a basis.
* **Theorem 3:** All bases for a specific vector space have the **same number of vectors**.
* **Theorem 4 (Dimension Theorem):** For subspaces $S_1$ and $S_2$:

$$\dim(S_1) + \dim(S_2) = \dim(S_1 + S_2) + \dim(S_1 \cap S_2)$$



---

## 5. Worked Examples: Finding Basis and Dimension

### Example A: A Plane in $\mathbb{R}^3$

**Given:** $S = \{(x_1, x_2, x_3) \in \mathbb{R}^3 : x_1 + x_2 - x_3 = 0\}$.

1. **Constraint:** $x_3 = x_1 + x_2$.
2. **Vector Form:** $(x_1, x_2, x_1 + x_2) = x_1(1,0,1) + x_2(0,1,1)$.
3. **Basis:** $\{(1,0,1), (0,1,1)\}$.
4. **Dimension:** $\dim(S) = 2$.

### Example B: Intersection of Subspaces

**Given:** $S$ (from above) and $W = \{(x,x,x) : x \in \mathbb{R}\}$.

1. **Solve simultaneously:** $x + x - x = 0 \Rightarrow x = 0$.
2. **Intersection:** Only the zero vector $(0,0,0)$.
3. **Dimension:** $\dim(S \cap W) = 0$.

### Example C: Subspaces in $\mathbb{R}^4$

**Given:** $S_1$ defined by $x_1+x_2-x_3+x_4=0$ and $x_1+x_2+x_3+x_4=0$.

1. Subtracting equations gives $2x_3 = 0 \Rightarrow x_3 = 0$.
2. Then $x_1 + x_2 + x_4 = 0 \Rightarrow x_4 = -(x_1 + x_2)$.
3. **Vector Form:** $(x_1, x_2, 0, -x_1 - x_2) = x_1(1,0,0,-1) + x_2(0,1,0,-1)$.
4. **Basis:** $\{(1,0,0,-1), (0,1,0,-1)\}$. $\dim(S_1) = 2$.

---

## 6. Significance in Machine Learning

* **PCA (Principal Component Analysis):** Finds an orthonormal basis where the first few vectors (principal components) capture the most variance.
* **Latent Spaces:** In Autoencoders, the bottleneck layer represents a lower-dimensional basis for the input data.
* **Sparsity:** Basis Pursuit algorithms try to represent data using as few basis vectors as possible.

---
