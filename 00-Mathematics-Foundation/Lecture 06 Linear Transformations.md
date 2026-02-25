I will ensure every core concept is reinforced with the exact examples and derivations provided in your text. Here are the structured notes for **Linear Transformations**, formatted for GitHub with complete LaTeX rendering and no omissions.

---

## Linear Transformations

*Essential Mathematics for ML — Structured Notes*

---

## 1. Fundamentals of Linear Transformations

A linear transformation is a mapping between vector spaces that preserves the operations of addition and scalar multiplication. They are used in Machine Learning to scale, rotate, reflect, and project data into spaces where it becomes more manageable or linearly separable.

### Formal Definition

Let $V$ and $W$ be vector spaces over a field $F$. A mapping $T: V \to W$ is a **linear transformation** if it satisfies two axioms for all $\mathbf{v}, \mathbf{v}_1, \mathbf{v}_2 \in V$ and $\alpha \in F$:

1. **Additivity:** $T(\mathbf{v}_1 + \mathbf{v}_2) = T(\mathbf{v}_1) + T(\mathbf{v}_2)$
2. **Homogeneity:** $T(\alpha \mathbf{v}) = \alpha T(\mathbf{v})$

> **Necessary Condition:** For any linear transformation, $T(\mathbf{0}) = \mathbf{0}$. If the origin does not map to zero, the transformation is not linear (it is likely affine).

---

## 2. Examples of Linearity Checks

### Example 1: Standard Linear Map

**Transformation:** $T: \mathbb{R}^2 \to \mathbb{R}^2, \quad T(x_1, x_2) = (x_1, x_1 + x_2)$

* **Additivity:** $T((x_1+y_1), (x_2+y_2)) = (x_1+y_1, x_1+y_1+x_2+y_2)$, which equals $T(\mathbf{v}_1) + T(\mathbf{v}_2)$.
* **Homogeneity:** $T(\alpha x_1, \alpha x_2) = (\alpha x_1, \alpha x_1 + \alpha x_2) = \alpha T(x_1, x_2)$.
* **Result:** Satisfies both; therefore, it is **Linear**.

### Example 2: Coordinate Shift

**Transformation:** $T: \mathbb{R}^3 \to \mathbb{R}^3, \quad T(x_1, x_2, x_3) = (x_2, x_1, 0)$

* **Result:** **Linear**. This acts as a reflection across the plane $x_1=x_2$ and a projection onto the $xy$-plane.

---

## 3. Geometrical Interpretation

Linear transformations alter the space while keeping the grid lines parallel and the origin fixed.

* **Scaling:** $T(x_1, x_2) = (2x_1, 2x_2)$ doubles the size of a square.
* **Stretching:** $T(x_1, x_2) = (x_1, 2x_2)$ elongates the space into a rectangle.
* **Projection:** $T(x_1, x_2) = (x_1, 0)$ flattens the space onto the $x$-axis.
* **Rotation:** Rotates a vector by angle $\theta$ using the rotation matrix:

$$T(\mathbf{x}) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$$



---

## 4. Relationship with Matrices

Every linear transformation can be represented as a matrix, and every matrix represents a linear transformation.

### From Transformation to Matrix

To build the matrix $A$, we map the standard basis vectors. The coefficients of $T(\mathbf{v}_j)$ form the $j$-th column of the matrix.

**Worked Example:**
$T(x_1, x_2) = (2x_1 - 7x_2, 4x_1 + 3x_2)$

* $T(1, 0) = (2, 4)$
* $T(0, 1) = (-7, 3)$
* **Matrix Representation:** $A = \begin{bmatrix} 2 & -7 \\ 4 & 3 \end{bmatrix}$

### From Matrix to Transformation

If $A = \begin{bmatrix} 2 & 1 \\ 4 & 3 \end{bmatrix}$, the transformation is $T(x_1, x_2) = (2x_1 + x_2, 4x_1 + 3x_2)$.

---

## 5. Null Space and Range

### Null Space (Kernel)

The set of input vectors that collapse to zero:


$$\text{Null}(T) = \{\mathbf{v} \in V : T(\mathbf{v}) = \mathbf{0}\}$$

* **Dimension:** Nullity($T$).

### Range (Image)

The set of all possible output vectors:


$$\text{Range}(T) = \{\mathbf{w} \in W : \exists \mathbf{v} \in V \text{ such that } T(\mathbf{v}) = \mathbf{w}\}$$

* **Dimension:** Rank($T$).

---

## 6. The Rank–Nullity Theorem

$$\boxed{\text{Rank}(T) + \text{Nullity}(T) = \dim(V)}$$

### Comprehensive Worked Example

**Transformation:** $T(x_1, x_2, x_3) = (x_1 - x_2 + x_3, x_2 - x_3, x_1, 2x_1 - 5x_2 + 5x_3)$

**1. Finding Range and Rank:**
Map the standard basis vectors:

* $T(1,0,0) = (1,0,1,2)$
* $T(0,1,0) = (-1,1,0,-5)$
* $T(0,0,1) = (1,-1,0,5)$
* *Observation:* The third vector is $-1 \times$ the second. Only two are independent.
* **Range:** $\text{span}\{(1,0,1,2), (-1,1,0,-5)\}$.
* **Rank(T):** 2.

**2. Finding Null Space and Nullity:**
Solve $T(\mathbf{x}) = \mathbf{0}$:

* $x_1 = 0$
* $x_2 - x_3 = 0 \implies x_2 = x_3$
* **Null Space:** $\text{span}\{(0,1,1)\}$.
* **Nullity(T):** 1.

**3. Verification:** $\text{Rank}(2) + \text{Nullity}(1) = 3$ (The dimension of input space $\mathbb{R}^3$).

---
