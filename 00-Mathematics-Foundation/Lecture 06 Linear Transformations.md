## Linear Transformations

*Essential Mathematics for ML — Structured Notes*

---

## 1. Fundamentals of Linear Transformations

### Motivation and Intuition
A linear transformation is a rule that takes a vector, morphs it, and produces a new vector. In Machine Learning, particularly in Neural Networks, every time a mathematical layer multiplies a weight matrix $W$ with an input vector $\mathbf{x}$, it applies a linear transformation. We use these transformations to stretch, rotate, reflect, and squeeze datasets until the target classes become linearly separable (easy to draw a line between).

### Formal Definition

Let $V$ and $W$ be vector spaces. A mapping $T: V \to W$ is a **linear transformation** if it satisfies two non-negotiable axioms for all vectors $\mathbf{v}$ and scalars $\alpha$:

1. **Additivity:** $T(\mathbf{v}_1 + \mathbf{v}_2) = T(\mathbf{v}_1) + T(\mathbf{v}_2)$
2. **Homogeneity:** $T(\alpha \mathbf{v}) = \alpha T(\mathbf{v})$

> **Necessary Condition:** $T(\mathbf{0}) = \mathbf{0}$. If the origin moves, the transformation is "affine", not linear. Neural network layers do $W\mathbf{x} + \mathbf{b}$. The addition of the bias $\mathbf{b}$ actually makes the layer an affine transformation, shifting the origin!

---

## 2. Examples of Linearity Checks

### Example 1: Standard Linear Map
$T(x_1, x_2) = (x_1, x_1 + x_2)$

* **Additivity:** $T(x+y) = T(x) + T(y)$.
* **Homogeneity:** $T(\alpha x_1, \alpha x_2) = (\alpha x_1, \alpha x_1 + \alpha x_2) = \alpha T(x_1, x_2)$.
* **Result:** **Linear.** 

### Example 2: Coordinate Shift
$T(x_1, x_2, x_3) = (x_2, x_1, 0)$

* **Result:** **Linear.** This reflects outputs across the $x_1=x_2$ plane and flattens onto $z=0$.

---

## 3. Geometrical Interpretation

Linear transformations keep grid lines parallel and evenly spaced, and leave the origin fixed.

* **Scaling:** $T(x_1, x_2) = (2x_1, 2x_2)$ expands the vector.
* **Projection:** $T(x_1, x_2) = (x_1, 0)$ deletes the second component, smashing the 2D space onto a 1D line.
* **Rotation:** Rotates space by $\theta$:

$$
T(\mathbf{x}) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

# A 90-degree rotation matrix
theta = np.pi / 2
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

# Transforming a point (1, 0)
x = np.array([1, 0])
rotated_x = np.dot(rotation_matrix, x) # Result: [0, 1]
```

---

## 4. Relationship with Matrices

**Every linear transformation is a matrix, and every matrix applies a linear transformation.** 

To build the matrix $A$, map the standard basis vectors (e.g., $(1,0)$ and $(0,1)$). The outputs become the columns of the matrix.

**Worked Example:**
$T(x_1, x_2) = (2x_1 - 7x_2, 4x_1 + 3x_2)$

* Transform basis $\mathbf{e}_1$: $T(1, 0) = (2, 4)$
* Transform basis $\mathbf{e}_2$: $T(0, 1) = (-7, 3)$
* **Matrix Representation:** 

$$
A = \begin{bmatrix} 2 & -7 \\ 4 & 3 \end{bmatrix}
$$

```python
# The transformation matrix A
A = np.array([[2, -7],
              [4,  3]])

x = np.array([1, 1])
# Transforming a point is just a matrix-vector dot product
T_x = A @ x  # Output: [-5, 7]
```

---

## 5. Null Space and Range

### Null Space (Kernel)

The set of inputs that collapse to zero (the origin). In ML, if a weight matrix has a large null space, it means the network is obliterating feature information, mapping distinct inputs to the exact same zero state.

$$
\text{Null}(T) = \{\mathbf{v} \in V : T(\mathbf{v}) = \mathbf{0}\}
$$

* **Dimension:** Nullity($T$).

### Range (Image)

The set of all possible outputs. If you multiply 3D vectors by a $3 \times 3$ matrix, but the range is a 2D plane, the matrix has "squashed" your space, losing one dimension of variance permanently.

$$
\text{Range}(T) = \{\mathbf{w} \in W : \exists \mathbf{v} \in V \text{ such that } T(\mathbf{v}) = \mathbf{w}\}
$$

* **Dimension:** Rank($T$).

---

## 6. The Rank–Nullity Theorem

$$
\boxed{\text{Rank}(T) + \text{Nullity}(T) = \dim(V)}
$$

This theorem proves information conservation. The dimensions you map onto (Rank) plus the dimensions you crush to zero (Nullity) must perfectly add up to the original dimensionality of your input space.

### Comprehensive Worked Example

**Transformation:** $T(x_1, x_2, x_3) = (x_1 - x_2 + x_3, x_2 - x_3, x_1, 2x_1 - 5x_2 + 5x_3)$
Input space $\mathbb{R}^3$, Output space $\mathbb{R}^4$.

**1. Range and Rank:**
Map the standard basis vectors:

* $T(1,0,0) = (1,0,1,2)$
* $T(0,1,0) = (-1,1,0,-5)$
* $T(0,0,1) = (1,-1,0,5)$

*Observation:* The third vector is strictly $-1 \times$ the second vector. They carry redundant information. Only two vectors are independent.

* **Range:** $\text{span}\{(1,0,1,2), (-1,1,0,-5)\}$.
* **Rank(T):** 2.

**2. Null Space and Nullity:**
Solve $T(\mathbf{x}) = \mathbf{0}$:

* $x_1 = 0$
* $x_2 - x_3 = 0 \implies x_2 = x_3$
* **Null Space:** Any vector of form $(0, a, a) \implies \text{span}\{(0,1,1)\}$.
* **Nullity(T):** 1.

**Verification:** $\text{Rank}(2) + \text{Nullity}(1) = \dim(\mathbb{R}^3) = 3$.

> **Check your intuition:** What kind of linear transformation has a rank of 0? *(Answer: The zero matrix. It crushes everything to the origin, meaning Nullity equals the entire input dimension.)*
