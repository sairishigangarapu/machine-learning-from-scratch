## Vectors in Machine Learning

*Essential Mathematics for ML — Structured Notes*

---

## 1. Fundamentals of Vectors

### Motivation and Intuition
Before diving into the math, let us look at why we need vectors. In Machine Learning, everything is represented as a list of numbers. If you are predicting the price of a house, you might care about 3 features:

1. Bedrooms: 3
2. Bathrooms: 2
3. Square Footage: 1500

We group these numbers into a single object: `[3, 2, 1500]`. This list is a **vector**. In a Deep Learning context, a single artificial neuron also takes such an input vector, multiplies it by a "weight" vector (representing the importance of each feature), and outputs a prediction.

### Formal Definition

In linear algebra, a vector is an element of a vector space characterized by magnitude and direction. A valid vector space satisfies two primary operations:

1. **Vector Addition**: Combining two vectors to produce a third.
2. **Scalar Multiplication**: Scaling a vector by a real number (scalar).

### Representation

In technical documentation, we distinguish between two orientations:

* **Row Vector:** $\mathbf{v} = [v_1, v_2, \dots, v_n]$

* **Column Vector:** 

$$
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}
$$

```python
import numpy as np

# A typical column vector or 1D array in numpy
house_vector = np.array([3, 2, 1500])
```

---

## 2. Vectors in $\mathbb{R}^n$

A vector $\mathbf{v} = (v_1, v_2, \dots, v_n)$ belongs to the $n$-dimensional Euclidean space $\mathbb{R}^n$ if all its components are real numbers.

* **$\mathbb{R}^2$:** Represented as $(x, y)$, visualized on a 2D plane.
* **$\mathbb{R}^3$:** Represented as $(x, y, z)$, visualized in 3D space.
* **Higher Dimensions:** Machine learning datasets easily reach thousands of dimensions. While $n > 3$ cannot be visualized, the algebraic properties remain identical.

### Geometric Interpretation

Geometrically, a vector is an arrow originating from the origin $(0, 0, \dots, 0)$. Each component represents the displacement along a specific axis. 

---

## Vector Algebra

### 3. Basic Operations

Operations are performed component-wise. Given $\mathbf{v} = (x_1, \dots, x_n)$ and $\mathbf{w} = (y_1, \dots, y_n)$:

* **Addition:** $\mathbf{v} + \mathbf{w} = (x_1 + y_1, x_2 + y_2, \dots, x_n + y_n)$
* **Subtraction:** $\mathbf{v} - \mathbf{w} = (x_1 - y_1, x_2 - y_2, \dots, x_n - y_n)$

```python
v = np.array([1, 2, 3])
w = np.array([4, 5, 6])

addition = v + w       # [5, 7, 9]
scalar_scaling = 3 * v # [3, 6, 9]
```

### 4. Dot Product (Inner Product)

The dot product is arguably the most important operation in Machine Learning. It measures how much two vectors "align" with each other.

$$
\mathbf{v} \cdot \mathbf{w} = \sum_{i=1}^{n} x_i y_i
$$

**The Neuron Analogy:** A biological neuron firing can be modeled mathematically as a dot product. An input vector $\mathbf{x}$ passes through a weight vector $\mathbf{w}$. The neuron computes $\mathbf{w} \cdot \mathbf{x} + b$ (where $b$ is a bias). A high dot product means the input strongly activated the neuron's learned pattern.

**Example in $\mathbb{R}^3$:**
$(1, 1, -1) \cdot (2, 3, 1) = (1 \times 2) + (1 \times 3) + (-1 \times 1) = 4$

```python
# The fundamental engine of neural networks
dot_product = np.dot(v, w)  # 1*4 + 2*5 + 3*6 = 32
```

### 5. Norm and Magnitude

The magnitude (or $L_2$ norm) of a vector represents its Euclidean length. It measures "how large" the feature vector is.

$$
\|\mathbf{v}\| = \sqrt{\mathbf{v} \cdot \mathbf{v}} = \sqrt{x_1^2 + x_2^2 + \dots + x_n^2}
$$

```python
magnitude = np.linalg.norm(v)  # sqrt(1^2 + 2^2 + 3^2) ≈ 3.74
```

### 6. Angles Between Vectors

The dot product and the geometric angle $\theta$ are related by:

$$
\cos \theta = \frac{\mathbf{v} \cdot \mathbf{w}}{\|\mathbf{v}\| \|\mathbf{w}\|}
$$

In NLP, this exact formula is known as **Cosine Similarity**, heavily used to measure the similarity between two word embeddings (like "king" and "queen").

---

## 7. Linear Combinations and Independence

### Linear Combination

Given a set of vectors $\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$, a linear combination is:

$$
\mathbf{u} = \alpha_1\mathbf{v}_1 + \dots + \alpha_k\mathbf{v}_k
$$

where $\alpha_i$ are scalars.

### Linear Independence and Failure Modes

A set of vectors is **Linearly Independent** if no vector can be expressed as a linear combination of the others. The only solution to $\alpha_1\mathbf{v}_1 + \dots + \alpha_n\mathbf{v}_n = \mathbf{0}$ is $\alpha_i = 0$.

**Why it matters in ML:** If a dataset has features that are linearly dependent (e.g., Feature A is distance in miles, Feature B is distance in kilometers), this is called **multicollinearity**. In classical Linear Regression, linearly dependent features cause the covariance matrix to become non-invertible (singular). This literal failure mode breaks the Normal Equation $\mathbf{w} = (X^T X)^{-1} X^T y$, crashing the model.

* **Linearly Dependent:** At least one vector is a redundant combination of the others.
* **Property:** In $\mathbb{R}^n$, any set with more than $n$ vectors is Linearly Dependent.

---

## 8. Orthogonality and Orthonormality

### Orthogonal Vectors

Two vectors are orthogonal if they are perpendicular, meaning they share no alignment. Their dot product is strictly zero:

$$
\mathbf{v}_i \cdot \mathbf{v}_j = 0, \quad \text{for } i \neq j
$$

### Orthonormal Vectors

A set is orthonormal if it satisfies two conditions:

1. Mutually orthogonal ($\mathbf{v}_i \cdot \mathbf{v}_j = 0$).
2. Unit magnitude ($\|\mathbf{v}\| = 1$).

Orthonormal matrices preserve distances and don't amplify gradients, making them highly desirable when initializing weights in deep neural networks to prevent exploding/vanishing gradients.

---

## 9. Vectors as Feature Vectors in Machine Learning

To summarize, datasets are matrices of stacked vectors:

* **Samples (Rows):** Each row represents an individual observation.
* **Features (Columns):** Each column represents an attribute.

**Feature Vector:** For sample $E_i$, $\mathbf{x}_i = [f_1, f_2, \dots, f_n]$.

> **Check your intuition:** If a dataset tracks coordinates of cars driving strictly straight along a 1D highway, but the data is recorded in 3D $(x, y, z)$, are the feature columns linearly independent? *(Answer: No. The motion is effectively 1D, meaning two columns are entirely predictable combinations of the other. The intrinsic dimensionality is 1.)*

---

## Prerequisites and Further Reading
- **Next:** [Lecture 02: Matrix Algebra](Lecture%2002%20Basics%20of%20Matrix%20Algebra.md) — Extends vector operations to matrices, enabling batch processing of data points
- **Related:** [Lecture 07: Norms](Lecture%2007%20Norms%20and%20Spaces.md) — Generalizes vector magnitude to higher-level loss functions and regularization
- **Related:** [Lecture 08: Orthogonal](Lecture%2008%20Orthogonal%20Complement%20and%20Projection%20Mapping.md) — Builds on dot product and angle concepts to projection and best-fit
