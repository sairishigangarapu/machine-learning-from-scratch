## 🎵 Introduction

In many ML algorithms, we represent **feature vectors** as **linear combinations** of certain vectors.  
These special vectors form a **basis** of the feature space.

We need a basis for tasks like:

- Dimensionality reduction
    
- Dictionary learning
    
- Wavelet transforms / orthonormal expansions
    
- PCA & SVD
    
- Sparse coding
    

So today we cover:

- What a **basis** is
    
- Finite & infinite dimensional vector spaces
    
- Examples
    
- Dimension
    
- Key theorems
    

---

# 1️⃣ Definition of a Basis

Let $V$ be a vector space over a field $F$.

A set $B = {v_1, v_2, \dots, v_n} \subseteq V$ is a **basis** of $V$ if:

### ✔ 1. The vectors are **linearly independent**

No vector can be written as a combination of the others.

### ✔ 2. The vectors **span** the space

Every $v \in V$ can be written as:

$$  
v = \alpha_1 v_1 + \alpha_2 v_2 + \dots + \alpha_n v_n,  
\quad \alpha_i \in F  
$$

If both conditions hold → this set is a **basis**.

---

# 2️⃣ Finite vs Infinite Dimensional Vector Spaces

### **Finite-dimensional**

If the basis has a **finite** number of vectors.

Example:

- $\mathbb{R}^2$ has dimension 2
    
- $\mathbb{R}^n$ has dimension $n$
    

### **Infinite-dimensional**

If basis uses **infinitely many vectors**.

Example:

- Space of all polynomials
    
- Space of continuous functions
    

---

# 3️⃣ Examples of Bases

## Example 1 — Basis of $\mathbb{R}^2$

$$  
(1,0),\ (0,1)  
$$

Linearly independent and span $\mathbb{R}^2$.

---

## Example 2 — Basis of $\mathbb{R}^3$

Standard basis:  
$$  
(1,0,0),\ (0,1,0),\ (0,0,1)  
$$

---

## Example 3 — Basis of $\mathbb{R}^n$

Standard basis:  
$$  
e_1 = (1,0,\dots,0), ;  
e_2 = (0,1,\dots,0), \dots, ;  
e_n = (0,0,\dots,1)  
$$

---

## Example 4 — Basis of $M_{2 \times 2}(\mathbb{R})$

One possible basis:  
$$
\begin{pmatrix}1 & 0 \\ 0 & 0\end{pmatrix},\;
\begin{pmatrix}0 & 1 \\ 0 & 0\end{pmatrix},\;
\begin{pmatrix}0 & 0 \\ 1 & 0\end{pmatrix},\;
\begin{pmatrix}0 & 0 \\ 0 & 1\end{pmatrix}


$$

Dimension = 4.

---

## Example 5 — Basis of $2 \times 2$ symmetric matrices

A symmetric matrix has the form:  
$$  
\begin{pmatrix}  
a & b \  
b & c  
\end{pmatrix}  
$$

Basis can be:  
$$  
\begin{pmatrix}1&0\\0&0\end{pmatrix},  
\begin{pmatrix}0&1\\1&0\end{pmatrix},  
\begin{pmatrix}0&0\\0&1\end{pmatrix}  
$$

Dimension = 3.

---

# 4️⃣ Dimension of a Vector Space

**Definition:**  
The dimension of $V$ = number of vectors in **any** basis of $V$.

### Examples:

- $\dim(\mathbb{R}^2) = 2$
    
- $\dim(\mathbb{R}^n) = n$
    
- $\dim(M_{m \times n}(\mathbb{R})) = mn$
    
- $\dim(\text{polynomials of degree }\le n) = n+1$
    
- $\dim(\text{all polynomials}) = \infty$
    

---

# 5️⃣ Important Theorems About Basis & Dimension

## ✔ Theorem 1 — “More than n vectors ⇒ dependent”

In an $n$-dimensional vector space:

> Any set of more than $n$ vectors is **linearly dependent**.

Explanation:  
You cannot fit more than $n$ linearly independent directions into an $n$-dimensional space.

---

## ✔ Theorem 2 — “You can extend LI sets to a basis”

If:

- $V$ is $n$-dimensional
    
- You have $k < n$ linearly independent vectors
    

Then:

> You can always add $(n-k)$ more independent vectors to form a basis.

Example:  
3 LI vectors in $\mathbb{R}^5$ → extend by 2 more vectors to get a basis.

---

## ✔ Theorem 3 — “All bases have the same size”

A finite-dimensional vector space:

> **Every basis has the SAME number of vectors**  
> (this number = the dimension)

---

## ✔ Theorem 4 — Dimension formula for subspaces

Let $S_1$ and $S_2$ be subspaces of $V$. Then:

$$  
\dim(S_1) + \dim(S_2)  
= \dim(S_1 + S_2) + \dim(S_1 \cap S_2)  
$$

This is the famous **dimension theorem** for subspaces.

---

# 6️⃣ Examples: Finding Basis & Dimension

## 💡 Example:

$$  
S = {(x_1,x_2,x_3) \in \mathbb{R}^3 : x_1 + x_2 - x_3 = 0}  
$$

Rewrite constraint:  
$$  
x_3 = x_1 + x_2  
$$

So vectors in $S$ look like:  
$$  
(x_1, x_2, x_1 + x_2)  
= x_1(1,0,1) + x_2(0,1,1)  
$$

Thus:

### Basis:

$$  
(1,0,1),\ (0,1,1)  
$$

### Dimension:

$$  
\dim(S) = 2  
$$

---

## 💡 Example:

$$  
W = {(x,x,x) : x \in \mathbb{R}}  
$$

All components equal → multiples of $(1,1,1)$.

### Basis:

$$  
(1,1,1)  
$$

### Dimension:

$$  
1  
$$

---

## 💡 Intersection Example

Find $\dim(S \cap W)$ for previous $S$ and $W$.

Solve simultaneously:

1. $x_1 + x_2 - x_3 = 0$
    
2. $x_1 = x_2 = x_3$
    

Substituting:

$x_1 + x_1 - x_1 = x_1 = 0$

Thus only solution:  
$$  
(0,0,0)  
$$

### Basis:

Empty set (because only zero vector)

### Dimension:

$$  
0  
$$

---

## 💡 Bigger Example — Subspaces in $\mathbb{R}^4$

Given:  
$$  
S_1 = {x \in \mathbb{R}^4 :  
\begin{cases}  
x_1 + x_2 - x_3 + x_4 = 0 \  
x_1 + x_2 + x_3 + x_4 = 0  
\end{cases}  
}  
$$


 $$
 S_{2}
= 
\left\{
(x_{1}, x_{2}, x_{3}, x_{4}) \in \mathbb{R}^{4}
\;\middle|\;
x_{1} - x_{2} - x_{3} + x_{4} = 0,\;
x_{1} + 2x_{2} - x_{4} = 0
\right\}

$$

Add equations → find $x_4$  
Subtract → find $x_3$  
Express vector as free variables → find basis.

Result:

### Basis of $S_1$:

$$  
(1,0,0,-1),\ (0,1,0,-1)  
$$

### Dimension:

$$  
2  
$$

Similarly,

### Basis of $S_2$:

$$  
(1,2,1,0),\ (1,1,1,2)  
$$

### Dimension:

$$  
2  
$$

Since:  
$$  
\dim(S_1) + \dim(S_2) = 4 = \dim(\mathbb{R}^4)  
$$

We get:  
$$  
\dim(S_1 \cap S_2) = 0  
$$

Meaning intersection contains only the zero vector.

---

# 🎤 Outro

In this lecture we covered:

- Basis
    
- Dimension
    
- Finite vs infinite dimensional spaces
    
- Standard bases
    
- Basis of matrices & symmetric matrices
    
- Subspace dimension theorem
    
- Multiple worked examples
    

Next lecture: **Linear Transformations**.

---
