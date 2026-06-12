## Basis and Dimension

*Essential Mathematics for ML — Structured Notes*

---

## 1. Definition of a Basis

### Motivation and Intuition
A $1024 \times 1024$ image dataset technically exists in a 1,000,000-dimensional space. However, most images of human faces don't use all those random dimensions—they share common structures like eyes, noses, and lighting gradients.

What if we could find a core set of, say, 50 "eigen-faces" such that blending them together in different proportions could perfectly recreate any face in our dataset? That set of 50 core images is a **Basis**. 

By finding a compact basis, Deep Learning models (like Autoencoders) can compress data from 1,000,000 dimensions down to a 50-dimensional "latent space", discarding the noise entirely and dramatically accelerating training.

### Formal Definition
A set of vectors $B = \{v_1, v_2, \dots, v_k\}$ forms a basis for a vector space $V$ if it satisfies two strict conditions:

1. **Linear Independence:** No vector in $B$ is a redundant copy or blend of the others. There is zero wasted information.
2. **Spanning Property:** You can reach *every single point* in $V$ by taking a linear combination of the vectors in $B$.

$$
\mathbf{v} = \alpha_1 \mathbf{v}_1 + \dots + \alpha_k \mathbf{v}_k
$$

---

## 2. Finite vs. Infinite Dimensional Spaces

The "size" of a vector space is perfectly defined by the number of vectors in its basis, a concept known as **Dimension**.

* **Finite-dimensional:** ML datasets lie here. If it takes 5 vectors to span the subspace, the dimension is 5.
* **Infinite-dimensional:** Theoretical ML (like kernel methods handling infinite feature spaces) uses spaces whose basis requires an infinite number of vectors (e.g., the space of all continuous functions).

```python
import numpy as np

# A dataset matrix where rows are linearly independent features
A = np.array([[1, 0, 0],
              [0, 1, 0]])

# The mathematical dimension of the space this matrix spans is its Rank
dimension = np.linalg.matrix_rank(A)  # Output: 2
```

---

## 3. Standard and Custom Bases

### Standard Bases
The default coordinate system we use every day.

* **In $\mathbb{R}^3$:** $e_1 = (1,0,0), e_2 = (0,1,0), e_3 = (0,0,1)$.

### Custom Bases
We can change our perspective. For a symmetric $2 \times 2$ matrix $\begin{bmatrix} a & b \\ b & c \end{bmatrix}$, the customized basis highlights its symmetries:

$$
\begin{bmatrix}1&0\\0&0\end{bmatrix}, \begin{bmatrix}0&1\\1&0\end{bmatrix}, \begin{bmatrix}0&0\\0&1\end{bmatrix}
$$

The dimension is 3, formally proving there are exactly 3 degrees of freedom ($a, b, c$) in a $2 \times 2$ symmetric matrix.

---

## 4. Fundamental Theorems

1. **Theorem 1:** In an $n$-dimensional space, any set with more than $n$ vectors is mathematically guaranteed to be **linearly dependent** (redundant).
2. **Theorem 2:** Any linearly independent set can be extended with more vectors to eventually form a full basis.
3. **Theorem 3:** Every valid basis for a specific vector space will *always* have the exact same number of vectors.

---

## 5. Worked Examples: Finding Basis and Dimension

### Example A: A Plane in $\mathbb{R}^3$
**Given:** Subspace $S$ defined by points where $x_1 + x_2 - x_3 = 0$.

1. **Constraint:** $x_3 = x_1 + x_2$.
2. **Vector Form:** Any point looks like $(x_1, x_2, x_1 + x_2)$.
3. **Split Variables:** $x_1(1,0,1) + x_2(0,1,1)$.
4. **Conclusion:** The basis is $\{(1,0,1), (0,1,1)\}$. Because there are 2 vectors, $\dim(S) = 2$. It's a 2D plane passing through the origin.

### Example C: Intersection of Subspaces
**Given:** $S_1$ defined by $x_1+x_2-x_3+x_4=0$ and $x_1+x_2+x_3+x_4=0$.

1. Subtracting the equations yields $2x_3 = 0 \Rightarrow x_3 = 0$.
2. Substitute back: $x_1 + x_2 + 0 + x_4 = 0 \Rightarrow x_4 = -x_1 - x_2$.
3. **Vector Form:** $(x_1, x_2, 0, -x_1 - x_2) = x_1(1,0,0,-1) + x_2(0,1,0,-1)$.
4. **Basis:** $\{(1,0,0,-1), (0,1,0,-1)\}$. $\dim(S_1) = 2$.

---

## 6. Significance in Deep Learning

* **Latent Spaces (Autoencoders):** A deep neural network encoder physically learns a new, customized, lower-dimensional basis that perfectly captures the underlying manifold of the training data.
* **PCA (Principal Component Analysis):** A rigid mathematical method to find an *orthonormal* basis where the first few vectors point precisely along the axes of highest dataset variance.
* **Sparsity:** Advanced regularization techniques (like L1 / Lasso) force neural networks to use as few basis vectors as possible, zeroing out the redundant ones to achieve model compression and interpretability.

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 04: Subspaces](Lecture%2004%20Vector%20Subspace.md) — Subspaces provide the geometric containers that bases span
- **Next:** [Lecture 06: Linear Transforms](Lecture%2006%20Linear%20Transformations.md) — Maps between spaces defined by bases; matrix representation depends on basis choice
- **Related:** [Lecture 27: Gram-Schmidt](Lecture%2027%20Gram%20Schmidt%20Process.md) — Converts any basis into an orthonormal basis for stable computation
