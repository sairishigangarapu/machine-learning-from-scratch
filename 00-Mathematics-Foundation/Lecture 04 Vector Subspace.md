## Vector Subspaces

*Essential Mathematics for ML — Structured Notes*

---

## 1. Definition of a Subspace

### Motivation and Intuition
Imagine a dataset with 3 features (living in a 3D room). What if two of those features are basically identical duplicates, and the third is just random noise? The data might formally exist in a 3D coordinate system, but geometrically, all the meaningful points are actually clustered tightly along a perfectly flat 2D plane passing through the origin of that room.

That flat 2D plane is a **Subspace**. Deep Learning and Dimensionality Reduction (like PCA) rely entirely on finding these lower-dimensional subspaces. If the data only truly "needs" a 2D subspace to exist, we can compress our 3D dataset down to 2D without losing information, drastically optimizing our Neural Networks.

### Formal Definition
Let $V$be a vector space over$\mathbb{R}$. A subset $S \subseteq V$is a **subspace** if$S$itself acts as a valid, self-contained vector space.

Geometrically, a subspace in$\mathbb{R}^n$is an **"origin-anchored flat slice."** It must be perfectly flat (linear) and it strictly must pass through the origin.

---

## 2. The Three-Point Subspace Criterion

Instead of checking all 8 vector space axioms again, a subset$S$ is mathematically guaranteed to be a subspace if it casually passes these three simple checks:

1. **Origin Check:** The zero vector is inside it ($\mathbf{0} \in S$).
2. **Closure under Addition:** If $\mathbf{x}, \mathbf{y} \in S$, then $\mathbf{x} + \mathbf{y}$stays inside$S$.
3. **Closure under Scalar Multiplication:** Scaling any vector in $S$keeps the result inside$S$.

> **Check your intuition:** Is a curved bowl shape touching the origin in 3D a subspace? *(Answer: No. While it contains the origin, scaling a vector pointing up the edge of the bowl will cause it to shoot straight through the curve into empty space, failing the scalar closure check. Subspaces must be flat.)*

---

## 3. Examples and Non-Examples

### Valid Subspaces
* **Trivial Subspaces:** For $\mathbb{R}^n$, the origin $\{\mathbf{0}\}$alone is a valid 0-D subspace. The entire$\mathbb{R}^n$room is a valid n-D subspace.
* **Homogeneous Planes:** The plane$x_1 + x_2 - x_3 = 0$is a subspace. It is flat and the point$(0,0,0)$perfectly solves the equation.

### Non-Examples (Failure Cases)
* **Affine Planes (Non-homogeneous):** The set$x_1 + x_2 + x_3 = 1$ is an affine plane. It hovers off the origin. It fails the zero check ($0+0+0 \neq 1$).
* **Unions:** Taking two intersecting 1D lines (two valid subspaces) and combining them. The combined set resembles an 'X'. 'X' is not a subspace because adding a vector from line 1 to line 2 results in a vector hovering between them, completely off the 'X' (failing closure).

---

## 4. Linear Span

The **span** of a set of vectors $S = \{\mathbf{v}_1, \dots, \mathbf{v}_n\}$is the set of literally every possible linear combination of those vectors.

$$\text{span}(S) = \{c_1 \mathbf{v}_1 + \dots + c_n \mathbf{v}_n : c_i \in \mathbb{R}\}$$

* **Property:** The span is mathematically defined as the **smallest possible subspace** that contains all vectors in$S$.

```python
import numpy as np

# Two linearly independent vectors in 3D
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])

# Any combination of v1 and v2 lies perfectly on the XY plane.
# The `span` of v1, v2 is exactly the XY plane (a 2D subspace of 3D).
combination = 3*v1 + 4*v2 # [3, 4, 0] -> Z is forever trapped at 0.
```

---

## 5. Matrix-Related Subspaces

Every $m \times n$matrix$A$acts as a linear transformation. We define four fundamental subspaces that govern exactly what$A$fundamentally *does* to data.

### 1. Column Space,$\text{Col}(A)$The span of the column vectors of$A$. It lives in $\mathbb{R}^m$.
**ML Connection:** This is identical to the **Range** of the transformation. It is the geometric space of all absolutely possible outputs the network layer $A$can produce.

### 2. Null Space,$N(A)$The set of all vectors$\mathbf{x}$that$A$crushes entirely to zero.
$$N(A) = \{ \mathbf{x} \in \mathbb{R}^n : A\mathbf{x} = \mathbf{0} \}$$
**ML Connection:** Any feature variance lying in the Null Space is permanently deleted by this matrix. It represents data loss.

### 3. Row Space,$\text{Row}(A)$The span of the rows of$A$(lives in$\mathbb{R}^n$). 

### 4. Left Null Space
The Null space of $A^T$.

**Deep Learning Failure Mode (Rank Deficiency):** If your matrix $A$is$1000 \times 1000$, but its Column Space is only a 10D subspace (Rank = 10), then 990 dimensions of your data input will fall straight into the Null Space and be annihilated to 0. A near-rank-deficient weight matrix causes catastrophic information bottlenecks in deep neural networks.
