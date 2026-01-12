## 🔹 Introduction

A **linear transformation** is one of the most important concepts in linear algebra, heavily used in:

- Machine Learning
    
- Computer Graphics & Animations
    
- Electrical & Electronics Engineering
    
- Data projections and feature transformations
    
- Calculus (derivatives are linear maps locally)
    

They are used to:

- Scale, rotate, reflect, and project data
    
- Map data into spaces where it becomes linearly separable
    
- Model real-world systems mathematically
    

---

## 🔹 Formal Definition

Let (V) and (W) be vector spaces over a field (F).

A mapping

$$[  
T: V \to W  
]
$$
is called a **linear transformation** if it satisfies:

### 1️⃣ Additivity

$$[  
T(\mathbf{v}_1 + \mathbf{v}_2) = T(\mathbf{v}_1) + T(\mathbf{v}_2)  
]
$$
### 2️⃣ Homogeneity (Scalar Multiplication)

 
$$T(\alpha \mathbf{v}) = \alpha T(\mathbf{v})  
$$
for all $$(\mathbf{v}, \mathbf{v}_1, \mathbf{v}_2 \in V)  and  (\alpha \in F).$$

➡️ If both hold, (T) is a **linear transformation**  
Also called: **linear map / linear mapping**

---

## 🔹 Examples of Linear Transformations

### ✅ Example 1

$$[  
T: \mathbb{R}^2 \to \mathbb{R}^2, \quad T(x_1, x_2) = (x_1, x_1 + x_2)  
]
$$
Check:

Additivity:

$$[  
T(\mathbf{v}_1 + \mathbf{v}_2) = T(\mathbf{v}_1) + T(\mathbf{v}_2)  
]$$

Homogeneity:

$$[  
T(\alpha \mathbf{v}) = \alpha T(\mathbf{v})  
]$$

✔ Satisfies both → Linear

---

### ✅ Example 2

$$[  
T: \mathbb{R}^3 \to \mathbb{R}^3, \quad T(x_1, x_2, x_3) = (x_2, x_1, 0)  
]$$

✔ Linear transformation

---

## 🔹 Geometrical Interpretation

Linear transformations can:

### 🔸 Scaling

$$[  
T(x_1, x_2) = (2x_1, 2x_2)  
]$$

➡ Doubles size of square

---

### 🔸 Stretching

$$[  
T(x_1, x_2) = (x_1, 2x_2)  
]
$$
➡ Rectangle formed

---

### 🔸 Projection

$$[  
T(x_1, x_2) = (x_1, 0)  
]$$

➡ Projects onto x-axis

---

### 🔸 Rotation

$$[  
T(x_1, x_2) =  
\begin{bmatrix}  
\cos\theta & -\sin\theta \\  
\sin\theta & \cos\theta  
\end{bmatrix}  
\begin{bmatrix}  
x_1 \  
x_2  
\end{bmatrix}  
]$$

➡ Rotates vector by angle $$(\theta)$$

---

## 🔹 Special Types

- **Linear Operator:** (T: V \to V)
    
- **Linear Functional:** (T: V \to F)
    

---

## 🔹 How to Check if a Map is Linear

Necessary condition:

$$[  
T(\mathbf{0}) = \mathbf{0}  
]
$$
If zero does not map to zero → ❌ Not linear

Also check:

- Additivity
    
- Homogeneity
    

---

## 🔹 Linear Transformations & Matrices

> ✅ Every linear transformation is a matrix  
> ✅ Every matrix represents a linear transformation

---

### 🔸 From Transformation → Matrix

Let:

- (V) have basis $$({\mathbf{v}_1, \dots, \mathbf{v}_n})$$
    
- (W) have basis $$({\mathbf{w}_1, \dots, \mathbf{w}_m})$$
    

Then:

$$[  
T(\mathbf{v}_j) = a_{1j}\mathbf{w}_1 + a_{2j}\mathbf{w}_2 + \dots + a_{mj}\mathbf{w}_m  
]$$

➡ The coefficients form the **j-th column** of the matrix.

---

### 🔸 Example

$$[  
T(x_1, x_2) = (2x_1 - 7x_2, ; 4x_1 + 3x_2)  
]$$

Matrix (standard basis):

$$[  
A =  
\begin{bmatrix}  
2 & -7 \\
4 & 3  
\end{bmatrix}  
]$$

---

### 🔸 From Matrix → Transformation

$$[  
A =  
\begin{bmatrix}  
2 & 1 \\  
4 & 3  
\end{bmatrix}  
]$$

$$[  
T(x_1, x_2) = (2x_1 + x_2, ; 4x_1 + 3x_2)  
]$$

---

## 🔹 Null Space and Range

Let $$(T: V \to W)$$

### 🔸 Null Space (Kernel)

$$[  
\text{Null}(T) = {\mathbf{v} \in V : T(\mathbf{v}) = \mathbf{0}}  
]$$

Subspace of (V)

---

### 🔸 Range (Image)

$$[  
\text{Range}(T) = {\mathbf{w} \in W : \exists \mathbf{v} \in V \text{ such that } T(\mathbf{v}) = \mathbf{w}}  
]$$

Subspace of (W)

---

### 🔸 Dimensions

- **Nullity(T)** = $$(\dim(\text{Null}(T)))$$
    
- **Rank(T)** = $$(\dim(\text{Range}(T)))$$
    

---

![[Pasted image 20260112223344.png]]

---
## 🔹 Rank–Nullity Theorem

$$[  
\boxed{\text{Rank}(T) + \text{Nullity}(T) = \dim(V)}  
]$$

---

## 🔹 Example (Range & Null Space)

$$[  
T(x_1, x_2, x_3) =  
(x_1 - x_2 + x_3, ; x_2 - x_3, ; x_1, ; 2x_1 - 5x_2 + 5x_3)  
]
$$
### Range

$$[  
T(1,0,0) = (1,0,1,2)  
]$$

$$[  
T(0,1,0) = (-1,1,0,-5)  
]$$

$$[  
T(0,0,1) = (1,-1,0,5)  
]$$

Third vector = (-1 \times) second → dependent

$$[  
\text{Range}(T) = \text{span}{(1,0,1,2), (-1,1,0,-5)}  
]$$

$$[  
\text{Rank}(T) = 2  
]$$

---

### Null Space

Solve:

$$[  
x_1 - x_2 + x_3 = 0  
]
$$
$$[  
x_2 - x_3 = 0  
]
$$
$$[  
x_1 = 0  
]$$

$$[  
2x_1 - 5x_2 + 5x_3 = 0  
]$$

➡ Solution:

$$[  
(x_1, x_2, x_3) = t(0,1,1)  
]
$$
$$[  
\text{Null}(T) = \text{span}{(0,1,1)}  
]$$

$$[  
\text{Nullity}(T) = 1  
]$$

✔ Rank + Nullity = (2 + 1 = 3)

---

## 🔹 Summary

- Linear transformations preserve vector structure
    
- They are represented by matrices
    
- Null space → what collapses to zero
    
- Range → what outputs are possible
    
- Rank + Nullity = Dimension of input space
    
