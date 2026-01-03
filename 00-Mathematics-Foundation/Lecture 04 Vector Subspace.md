## 🎵 Introduction

In the last lecture, we learned:

- What vector spaces are
    
- Examples of vector spaces
    

In this lecture, we cover:

- **Vector subspaces**
    
- **Examples**
    
- **Key properties**
    
- **Matrix-related subspaces**
    

The field is $\mathbb{R}$ (real numbers), as usual in machine learning.

---

# 1️⃣ Definition of a Subspace

Let $\mathbb{R}^n$ be a vector space over $\mathbb{R}$.

A subset $S \subseteq \mathbb{R}^n$ is called a **subspace** of $\mathbb{R}^n$ if:

### ✔ $S$ is itself a vector space

Meaning: the vectors of $S$ satisfy **all 6 vector space axioms** under the **same operations** as $\mathbb{R}^n$.


> [!important] Vector Space vs Subspace (Layman Analogy)
> **Vector Space** = an entire infinite room where all movement is allowed.  
> **Subspace** = a perfectly flat surface (line, plane, or hyperplane) *inside that room*  
> that **must pass through the origin** and follow all the same vector rules.
>
> - Subspace is not “smaller space shaped like a cube.”  
> - It’s an “origin-anchored flat slice” of the big space.  
> - Scaling or adding vectors keeps you inside that surface — that's why it counts.

---

# 2️⃣ Alternate Subspace Definition (Much Easier)

A subset $S \subseteq \mathbb{R}^n$ is a subspace if ALL the following hold:

1. **Zero vector is in $S$**  
    $$ 0 \in S $$
    
2. **Closed under vector addition**  
    If $x, y \in S$ then  
    $$ x + y \in S $$
    
3. **Closed under scalar multiplication**  
    If $x \in S$ and $\alpha \in \mathbb{R}$, then  
    $$ \alpha x \in S $$
    

This is the version used in most proofs.

---

# 3️⃣ Important Theorem About Subspaces

Let $W \subseteq \mathbb{R}^n$.  
Then $W$ is a subspace **iff**:

- $W \neq \varnothing$
    
- For all scalars $a, b \in \mathbb{R}$ and vectors $u, v \in W$:  
    $$ au + bv \in W $$
    

This single condition implies **all 6 vector space axioms**, making it the most convenient criterion.

---

# 4️⃣ Examples of Subspaces

## **Example 1 — Trivial subspaces**

If $V$ is a vector space:

- $S = {0}$
    
- $S = V$
    

Both are automatically subspaces → **trivial subspaces**.

---

## **Example 2 — Symmetric $3 \times 3$ matrices**

Let  
$$ V = M_{3 \times 3}(\mathbb{R}) $$

The subset:

> All symmetric $3 \times 3$ matrices

is a subspace because:

- Zero matrix is symmetric
    
- Sum of symmetric matrices is symmetric
    
- Scalar multiple of a symmetric matrix is symmetric
    

---

## **Example 3 — Plane in $\mathbb{R}^3$: $x_1 + x_2 - x_3 = 0$**

Define:  
$$  
S = {(x_1,x_2,x_3) \in \mathbb{R}^3 : x_1 + x_2 - x_3 = 0}  
$$

### ✔ Zero vector check

$0 + 0 - 0 = 0$ → $0 \in S$

### ✔ Closure under scalar multiplication

Take $a \in \mathbb{R}$:  
$$  
x_1 + x_2 - x_3 = 0 \Rightarrow  
a x_1 + a x_2 - a x_3 = a(0) = 0  
$$

### ✔ Closure under addition

If:  
$$  
x_1 + x_2 - x_3 = 0,\qquad  
y_1 + y_2 - y_3 = 0  
$$

Then:  
$$  
(x_1+y_1) + (x_2+y_2) - (x_3+y_3) = 0  
$$

Thus **$S$ is a subspace of $\mathbb{R}^3$**.

---

## **Example 4 — Line where all components are equal: $x_1 = x_2 = x_3$**

Define:  
$$  
S = { (x,x,x) : x \in \mathbb{R} }  
$$

Contains $(0,0,0)$.

Closed under:

- Addition:  
    $(x,x,x) + (y,y,y) = (x+y, x+y, x+y)$
    
- Scalar multiplication:  
    $a(x,x,x) = (ax,ax,ax)$
    

So it is a subspace.

---

## ❌ **Non-Example — $x_1 + x_2 + x_3 = 1$ is NOT a subspace**

Define:  
$$  
S = { (x_1,x_2,x_3) : x_1 + x_2 + x_3 = 1 }  
$$

Check zero vector:  
$$  
0 + 0 + 0 = 0 \neq 1  
$$

Zero vector is missing → **not a subspace**.

---

# 5️⃣ Geometric Meaning of Subspaces in $\mathbb{R}^3$

A subspace of $\mathbb{R}^3$ can only be:

1. ${0}$
    
2. A **line** through the origin
    
3. A **plane** through the origin
    
4. The entire $\mathbb{R}^3$
    

These are the only possibilities.

---

# 6️⃣ Intersection vs Union of Subspaces

## ✔ Intersection

If $S_1$ and $S_2$ are subspaces:

$$ S_1 \cap S_2 \text{ is always a subspace} $$

## ❌ Union (usually NOT a subspace)

$$ S_1 \cup S_2 \text{ is NOT necessarily a subspace} $$

The **only time** union is a subspace is when:

- One is contained in the other  
    $$ S_1 \subseteq S_2 \quad\text{or}\quad S_2 \subseteq S_1 $$
    

---

# 7️⃣ Example of Intersection

Let

- $S_1 = {x_1+x_2-x_3=0}$
    
- $S_2 = {x_1 = x_2 = x_3}$
    

Solve both conditions simultaneously.

From $S_2$:  
$$ x_1 = x_2 = x_3 = t $$

Plug into $S_1$:  
$$ t + t - t = t = 0 $$

Thus $t=0$ → only vector in intersection is:

$$ (0,0,0) $$

So:  
$$ S_1 \cap S_2 = {0} $$

A trivial subspace.

---

# 8️⃣ Linear Span

Let $V$ be a vector space.

Let  
$$ S = {v_1, v_2, \dots, v_n} \subseteq V $$

The **linear span** of $S$ is:  
$$  
\text{span}(S) = {c_1 v_1 + c_2 v_2 + \dots + c_n v_n : c_i \in \mathbb{R}}  
$$

This is **the set of all linear combinations** of the vectors.

### Examples:

### Example 1

$S = {(1,0), (0,1)}$ in $\mathbb{R}^2$

Span:  
$$  
c_1(1,0) + c_2(0,1) = (c_1, c_2)  
$$

This is:  
$$ \mathbb{R}^2 $$

---

### Example 2

$S = {(1,2,0), (1,1,-1)}$

Compute:  
$$  
c_1(1,2,0) + c_2(1,1,-1)  
= (c_1+c_2,, 2c_1+c_2,, -c_2)  
$$

That set can be rewritten as:  
$$  
{(x_1,x_2,x_3) : 2x_1 - x_3 = x_2}  
$$

This is a plane through origin → subspace of $\mathbb{R}^3$.

---

## ✔ Important result

$\text{span}(S)$ is **always** a subspace of $V$.

Also:

> It is the **smallest** subspace containing $S$.

---

# 9️⃣ Matrix-Related Subspaces

Let $A$ be an $m \times n$ matrix.

### 1. **Row space**

$$ \text{Row}(A) = \text{span of all rows of } A $$  
A subspace of $\mathbb{R}^n$.

---

### 2. **Column space**

$$ \text{Col}(A) = \text{span of all columns of } A $$  
A subspace of $\mathbb{R}^m$.

Column space = **range** of the linear transformation $Ax$.

---

### 3. **Null space**

$$  
N(A) = { x \in \mathbb{R}^n : Ax = 0 }  
$$

A subspace of $\mathbb{R}^n$.

---

### 4. **Range space**

$$  
\text{Range}(A) = { Ax : x \in \mathbb{R}^n }  
$$

But:  
$$  
\text{Range}(A) = \text{Col}(A)  
$$

These four subspaces are critical in ML (PCA, SVD, dimensionality reduction).

---
