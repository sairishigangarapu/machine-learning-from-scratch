
# 🧮 Matrix Algebra for Machine Learning

_Essential Mathematics — Structured Notes_

---

## 📌 1. What is a Matrix?

A **matrix** is a **2-dimensional array of scalars** (real numbers in our context).

### Notation

- A matrix with **m rows** and **n columns** is an **m × n** matrix.
    
- Element at row _i_, column _j_ is denoted **aᵢⱼ**.
    

### Examples

|Matrix|Meaning|
|---|---|
|`3 × 3`|Square matrix|
|`2 × 3`|Rectangular matrix|
|`3 × 1`|Column vector|
|`1 × 3`|Row vector|

---

# 🧩 2. Special Matrices

## ✔️ Diagonal Matrix

All **off-diagonal** elements are zero.

```
[d₁ 0  0
 0  d₂ 0
 0  0  d₃]
```

## ✔️ Zero Matrix

All elements = 0.

## ✔️ Upper Triangular Matrix

Entries **below** main diagonal = 0.

## ✔️ Lower Triangular Matrix

Entries **above** main diagonal = 0.

## ✔️ Identity Matrix (Iₙ)

Diagonal matrix with all diagonal entries = 1.

```
I₂ = [1 0
      0 1]
```

Identity matrices are always **square**.

---

# ➕ 3. Matrix Equality

Two matrices A and B are equal if:

1. They have the **same size**
    
2. All corresponding entries match
    
    ```
    aᵢⱼ = bᵢⱼ for all i,j
    ```
    

---

# ➕➖ 4. Matrix Addition & Subtraction

Defined **only when** matrices have the same dimensions.

Given A and B (both m × n):

```
(A ± B)ᵢⱼ = aᵢⱼ ± bᵢⱼ
```

### Properties

- Commutative: A + B = B + A
    
- Associative: A + (B + C) = (A + B) + C
    
- Subtraction is **not** commutative.
    

---

# ✖️ 5. Scalar Multiplication

Given scalar α and matrix A:

```
(αA)ᵢⱼ = α · aᵢⱼ
```

Just multiply every entry.

---

# ✖️✖️ 6. Matrix Multiplication (Most Important)

Matrix multiplication A·B is defined only when:

```
A is m × n  
B is n × p  
```

Result C = A·B is:

```
m × p
```

### Entry Formula

```
cᵢⱼ = (i-th row of A) ⋅ (j-th column of B)
```

(dot product)

### Notes

- AB may exist but BA may NOT exist
    
- Even if both exist, **AB ≠ BA** in general
    

### Properties

- Associative: A(BC) = (AB)C
    
- Distributive: A(B + C) = AB + AC
    
- Identity: AI = IA = A
    

### Non-intuitive Matrix Facts

- AB = 0 **does NOT imply** A = 0 or B = 0
    
- If AB = AC, it does **not** mean B = C
    

---

# 🔁 7. Transpose of a Matrix (Aᵀ)

Swap rows ↔ columns.

If A is m × n, then Aᵀ is n × m.

### Examples

```
A = [1 3
     2 1
     1 -1]

Aᵀ = [1 2 1
      3 1 -1]
```

### Properties

- (A + B)ᵀ = Aᵀ + Bᵀ
    
- (AB)ᵀ = Bᵀ Aᵀ
    
- (αA)ᵀ = αAᵀ
    
- (Aᵀ)ᵀ = A
    

---

# 🧮 8. Determinant (Only for Square Matrices)

Det(A) = scalar value.

Example (2×2):

```
A = [a b
     c d]

det(A) = ad – bc
```

Interpretation:

- det(A) = 0 → A is **singular** (no inverse)
    
- det(A) ≠ 0 → **invertible**
    

---

# 🔄 9. Inverse of a Matrix (A⁻¹)

Defined only for **square** & **non-singular** matrices.

A⁻¹ is such that:

```
A A⁻¹ = A⁻¹ A = I
```

### Key Properties

- (AB)⁻¹ = **B⁻¹ A⁻¹**
    
- (A⁻¹)⁻¹ = **A**
    
- (Aᵀ)⁻¹ = **(A⁻¹)ᵀ**
    
- (kA)⁻¹ = **(1/k) A⁻¹**
    
- det(A) = 0 → **A⁻¹ does NOT exist**
    

---

## 🔧 Adjugate (adj(A))

```
A⁻¹ = adj(A) / det(A)
```

### How to Compute adj(A)

1. **Find cofactors** of all entries
    
    - Cᵢⱼ = (−1)⁽ⁱ⁺ʲ⁾ · det(minor of aᵢⱼ)
        
2. **Form the cofactor matrix**
    
    - Put all Cᵢⱼ in their original positions
        
3. **Transpose the cofactor matrix**
    
    - adj(A) = (cofactor matrix)ᵀ
        

---

### Properties of adj(A)

- adj(AB) = **adj(B) adj(A)**
    
- adj(Aᵀ) = **(adj(A))ᵀ**
    
- adj(kA) = **kⁿ⁻¹ adj(A)** (n = order of matrix)
    

### Fundamental Identity

```
A · adj(A) = adj(A) · A = det(A) I
```

If det(A) = 1 → **A⁻¹ = adj(A)**

---

## 🧭 Orthogonal Matrices (Q)

Columns/rows are **orthonormal vectors**.

Definition:

```
Qᵀ Q = QQᵀ = I
```

Hence:

```
Q⁻¹ = Qᵀ
```

Properties:

- det(Q) = ±1
    
- Orthogonal ⇒ preserves **length, angle, dot product**
    

---

# 🐍 10. Python (NumPy) Commands

```python
import numpy as np

# Define a matrix
P = np.array([[1,7], [2,1], [3,2]])  # 3×2

# Addition
C1 = np.add(P, Q)

# Subtraction
C2 = np.subtract(P, Q)

# Matrix multiplication
C3 = np.matmul(P, R)

# Determinant
det = np.linalg.det(M)

# Inverse
invM = np.linalg.inv(M)
```

---

# 🤖 11. Why Matrices Matter in ML

Matrices represent:

- **Datasets**
    
    - rows → samples
        
    - columns → features
        
- **Weight matrices** in neural networks
    
- **Linear transformations**
    
- **Covariance matrices**
    
- **Distance metrics & projections**
    

Matrix operations power:

- PCA
    
- Linear Regression
    
- Neural Networks
    
- SVD
    
- Optimization and gradient steps
    

---
