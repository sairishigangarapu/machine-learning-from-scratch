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
; a_{21}, a_{22}, \dots, a_{2n},  
; \dots,  
; a_{m1}, a_{m2}, \dots, a_{mn})  
$$

---

## Column-major Vectorization

$$  
\text{vec}_{c}(A) =  
(a_{11}, a_{21}, \dots, a_{m1},  
; a_{12}, a_{22}, \dots, a_{m2},  
; \dots,  
; a_{1n}, a_{2n}, \dots, a_{mn})  
$$

---

## 2×2 Example Matrix

$$  
A =  
\begin{pmatrix}  
1 & 2 \\  
3 & 4  
\end{pmatrix}  
$$

Row-major:

$$  
(1, 2, 3, 4)  
$$

Column-major:

$$  
(1, 3, 2, 4)  
$$

---

## Basis Representation of a 2×2 Matrix

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

## Matrix Space as a Vector Space

$$  
M_{m \times n}(\mathbb{R}) \cong \mathbb{R}^{mn}  
$$

---

Here is the **same section with PERFECT, clean LaTeX**, fully Obsidian-ready and without duplication or broken symbols.

---

## **Geometric Interpretation**

### 💡 Idea

A matrix
$$
[  
A \in \mathbb{R}^{m \times n}  
]
$$
is normally thought of as a **2D object**.  
But 
$$
[  
\text{vec}(A)  
]  
$$is a point in **(mn)-dimensional space**.

---

### 🔭 Visualization

- A **2D grid** (matrix) → a **single point in a very high-dimensional space**.
    
- Each cell $$(a_{ij})$$ becomes one coordinate in $$(\mathbb{R}^{mn})$$.
    
- The geometry of matrix space = the geometry of a vector space of dimension (mn).
    

---

### ✨ Why this matters?

- Enables **inner products**, **norms**, **gradients**, **distance measures**, etc., on matrices
    
- Used in ML when flattening matrices (e.g., CNN kernels, linear layers)
    

---

### 🧭 Example: geometric view

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
So the matrix corresponds to the point
$$
(1, 3, 2, 4)
$$

in **4-dimensional space**.

----

# 📐 ASCII Diagram — How an (m \times n) Matrix Turns Into a Vector

### **Matrix $$(A \in \mathbb{R}^{m \times n})$$

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

## 🔽 **Vectorization (vec)**

Stack columns **top → bottom**, **left → right**

```
        vec(A)
        ▼
      ┌────────────────┐
      │   a11          │  ← first column, top to bottom
      │   a21          │
      │   ...          │
      │   am1          │
      │   a12          │  ← second column
      │   a22          │
      │   ...          │
      │   am2          │
      │    ...         │
      │   a1n          │  ← nth column
      │   ...          │
      │   amn          │
      └────────────────┘
```

---

# 🎯 **Mini-Example (2×3 matrix)**

Matrix:

```
A =
┌─────┬─────┬─────┐
│  1  │  2  │  3  │
│  4  │  5  │  6  │
└─────┴─────┴─────┘
```

Vectorization step-by-step:

```
Column 1 → 1, 4
Column 2 → 2, 5
Column 3 → 3, 6
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

---

