## Matrix Definition

$$  
A =  
\begin{pmatrix}  
a_{11} & a_{12} & \dots & a_{1n} \\  
a_{21} & a_{22} & \dots & a_{2n} \\  
\vdots & \vdots & \ddots & \vdots \\  
a_{m1} & a_{m2} & \dots & a_{mn}  
\end{pmatrix}  
$$

---

## Row-major Vectorization

$$  
\text{vec}_{r}(A) =  
(a_{11}, a_{12}, \dots, a_{1n},  
a_{21}, a_{22}, \dots, a_{2n},  
\dots,  
a_{m1}, a_{m2}, \dots, a_{mn})  
$$

---

## Column-major Vectorization

$$  
\text{vec}_{c}(A) =  
(a_{11}, a_{21}, \dots, a_{m1},  
; a_{12}, a_{22}, \dots, a_{m2},  
\dots,  
; a_{1n}, a_{2n}, \dots, a_{mn})  
$$

```python
import numpy as np

A = np.array([[1, 2], 
              [3, 4]])

# By default, NumPy uses Row-major (C-style) flattening
row_major = A.flatten()         # [1, 2, 3, 4]

# Can be forced to Column-major (Fortran-style)
col_major = A.flatten(order='F') # [1, 3, 2, 4]
```

---

## Basis Representation of a 2x2 Matrix

$$  
A =  
\begin{pmatrix}  
a & b \\  
c & d  
\end{pmatrix}  
$$

Expanded using basis matrices:

$$  
A =  
a\begin{pmatrix}1 & 0 \\ 0 & 0\end{pmatrix}  
+  
b\begin{pmatrix}0 & 1 \\ 0 & 0\end{pmatrix}  
+  
c\begin{pmatrix}0 & 0 \\ 1 & 0\end{pmatrix}  
+  
d\begin{pmatrix}0 & 0 \\ 0 & 1\end{pmatrix}  
$$

---

## Geometric Interpretation

### Motivation and Intuition

A matrix $A \in \mathbb{R}^{m \times n}$is typically thought of as a 2D grid of numbers. But in Deep Learning, we frequently pass images (which are 2D grids of pixels) into standard fully-connected Neural Networks. A basic feedforward network only accepts 1-dimensional vectors. 

To solve this, we "flatten" or "vectorize" the matrix. This conceptually transforms our 2D image matrix into a single point in an extremely high-dimensional space. The vectorization operator connects the geometry of a matrix space to the geometry of a vector space of dimension$mn$.

$$  
M_{m \times n}(\mathbb{R}) \cong \mathbb{R}^{mn}  
$$

### Visualization

- A 2D grid (matrix) -> a single point in a very high-dimensional space.
- Each cell $a_{ij}$becomes one coordinate in$\mathbb{R}^{mn}$.
- The geometry of matrix space is mathematically identical to the geometry of a vector space of dimension $mn$.

### Deep Learning Connection: Why this matters

- **Flattening Layers:** Used inherently in Machine Learning when flattening matrices. For example, flattening the output of a Convolutional Neural Network (CNN) before passing it into a linear dense layer. 
- **Distance Metrics:** By treating matrices as vectors, we can instantly apply vector operations like inner products, norms, and distance measures to whole images. We can measure how "far apart" two images are simply by taking the Euclidean distance between their vectorized forms.

---

### Example: Geometric view

If
$$
A =
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
$$
then
$$
\text{vec}(A) =
\begin{bmatrix}
1 \\ 3 \\ 2 \\ 4
\end{bmatrix}
= (1, 3, 2, 4)^{T}
$$

So the matrix corresponds to the point $(1, 3, 2, 4)$in 4-dimensional space.

----

## ASCII Diagram - How an (m x n) Matrix Turns Into a Vector

**Matrix$A \in \mathbb{R}^{m \times n}$**

```
      Column 1      Column 2        ...       Column n
     ┌─────────┬─────────┬───────┬─────────┐
Row 1│ a11     │ a12     │  ...  │ a1n     │
Row 2│ a21     │ a22     │  ...  │ a2n     │
 ... │  ...    │  ...    │  ...  │  ...    │
Row m│ am1     │ am2     │  ...  │ amn     │
     └─────────┴─────────┴───────┴─────────┘
```

---

**Vectorization (vec)**

Stack columns top to bottom, left to right (Column-major format, often used in theoretical linear algebra, though standard Python implementations default to row-major).

```
        vec(A)
        |
      ┌────────────────┐
      │   a11          │  <- first column, top to bottom
      │   a21          │
      │   ...          │
      │   am1          │
      │   a12          │  <- second column
      │   a22          │
      │   ...          │
      │   am2          │
      │    ...         │
      │   a1n          │  <- nth column
      │   ...          │
      │   amn          │
      └────────────────┘
```

---

## Mini-Example (2x3 matrix)

Matrix:

```
A =
┌─────┬─────┬─────┐
│  1  │  2  │  3  │
│  4  │  5  │  6  │
└─────┴─────┴─────┘
```

Vectorization step-by-step (Column-major):

```
Column 1 -> 1, 4
Column 2 -> 2, 5
Column 3 -> 3, 6
```

Final vector:

```
vec(A) =
┌───┐
│ 1 │
│ 4 │
│ 2 │
│ 5 │
│ 3 │
│ 6 │
└───┘
```
