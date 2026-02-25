## Vectors in Machine Learning

*Essential Mathematics for ML — Structured Notes*

---

## 1. Fundamentals of Vectors

In the context of Machine Learning and Linear Algebra, a vector is a mathematical object characterized by magnitude (length) and direction. Formally, a vector is an element of a vector space, which is a collection of objects that satisfies two primary operations:

1. **Vector Addition**: Combining two vectors to produce a third.
2. **Scalar Multiplication**: Scaling a vector by a real number (scalar).

### Representation

In technical documentation, we distinguish between two primary orientations:

* **Row Vector:** $\mathbf{v} = [v_1, v_2, \dots, v_n]$
* **Column Vector:** 
$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$



---

## 2. Vectors in $\mathbb{R}^n$

A vector $\mathbf{v} = (v_1, v_2, \dots, v_n)$ belongs to the $n$-dimensional Euclidean space $\mathbb{R}^n$ if all its components are real numbers.

* **$\mathbb{R}^2$:** Represented as $(x, y)$, visualized on a plane.
* **$\mathbb{R}^3$:** Represented as $(x, y, z)$, visualized in 3D space.
* **Higher Dimensions:** While $n > 3$ cannot be visualized, the algebraic properties remain consistent across dimensions.

### Geometric Interpretation

Geometrically, a vector is typically viewed as an arrow originating from the origin $(0, 0, \dots, 0)$. Each component represents the displacement along a specific axis.

---

## Vector Algebra

### 3. Basic Operations

Operations are performed component-wise. Given $\mathbf{v} = (x_1, \dots, x_n)$ and $\mathbf{w} = (y_1, \dots, y_n)$:

* **Addition:** $\mathbf{v} + \mathbf{w} = (x_1 + y_1, x_2 + y_2, \dots, x_n + y_n)$
* **Subtraction:** $\mathbf{v} - \mathbf{w} = (x_1 - y_1, x_2 - y_2, \dots, x_n - y_n)$

### 4. Dot Product (Inner Product)

The dot product of two vectors in $\mathbb{R}^n$ results in a scalar value:


$$\mathbf{v} \cdot \mathbf{w} = \sum_{i=1}^{n} x_i y_i$$

**Example in $\mathbb{R}^3$:**
$(1, 1, -1) \cdot (2, 3, 1) = (1 \times 2) + (1 \times 3) + (-1 \times 1) = 2 + 3 - 1 = 4$

### 5. Norm and Magnitude

The magnitude (or $L_2$ norm) of a vector represents its Euclidean length:


$$\|\mathbf{v}\| = \sqrt{\mathbf{v} \cdot \mathbf{v}} = \sqrt{x_1^2 + x_2^2 + \dots + x_n^2}$$

**Example:** For $\mathbf{v} = (1, -1, 2)$, $\|\mathbf{v}\| = \sqrt{1^2 + (-1)^2 + 2^2} = \sqrt{6}$.

### 6. Angles Between Vectors

The relationship between the dot product and the geometric angle $\theta$ is defined by:


$$\cos \theta = \frac{\mathbf{v} \cdot \mathbf{w}}{\|\mathbf{v}\| \|\mathbf{w}\|}$$

Consequently, the angle can be determined using the arccosine:


$$\theta = \arccos \left( \frac{\mathbf{v} \cdot \mathbf{w}}{\|\mathbf{v}\| \|\mathbf{w}\|} \right)$$

---

## 7. Linear Combinations and Independence

### Linear Combination

Given a set of vectors $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\}$, a linear combination is defined as:


$$\mathbf{u} = \alpha_1\mathbf{v}_1 + \alpha_2\mathbf{v}_2 + \dots + \alpha_k\mathbf{v}_k$$


where $\alpha_i$ are scalar coefficients.

### Linear Independence (LI)

A set of vectors is Linearly Independent if the only solution to the following equation is the trivial solution ($\alpha_i = 0$ for all $i$):


$$\alpha_1\mathbf{v}_1 + \alpha_2\mathbf{v}_2 + \dots + \alpha_n\mathbf{v}_n = \mathbf{0}$$

> **Key Concept:** In a linearly independent set, no vector can be expressed as a linear combination of the others.

### Linear Dependence (LD)

A set is Linearly Dependent if at least one vector can be expressed as a linear combination of the others.

* **Property:** In $\mathbb{R}^n$, any set containing more than $n$ vectors is Linearly Dependent.
* **Property:** Any set containing the zero vector is Linearly Dependent.

---

## 8. Orthogonality and Orthonormality

### Orthogonal Vectors

Two vectors are orthogonal if they are perpendicular, resulting in a dot product of zero:


$$\mathbf{v}_i \cdot \mathbf{v}_j = 0, \quad \text{for } i \neq j$$

### Orthonormal Vectors

A set is orthonormal if it satisfies two conditions:

1. All vectors are mutually orthogonal.
2. Each vector has a unit magnitude ($\|\mathbf{v}\| = 1$).

---

## 9. Vectors as Feature Vectors in Machine Learning

In Machine Learning datasets, vectors serve as the fundamental unit of data representation:

* **Samples (Rows):** Each row represents an individual data point or observation.
* **Features (Columns):** Each column represents a specific attribute.

**Feature Vector:** For a specific sample $E_i$, the feature vector is represented as $\mathbf{x}_i = [f_1, f_2, \dots, f_n]$.

---

## 10. Implementation via NumPy

```python
import numpy as np

# Vector initialization
v = np.array([1, -1, 2])
w = np.array([2, 5, 2])

# Basic Operations
addition = v + w               # Vector Addition
subtraction = v - w            # Vector Subtraction
scalar_scaling = 3 * v         # Scalar Multiplication

# Linear Algebra Metrics
magnitude = np.linalg.norm(v)  # L2 Norm (Magnitude)
dot_product = np.dot(v, w)     # Dot Product

```

---
