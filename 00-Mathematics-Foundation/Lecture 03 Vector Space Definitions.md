## 🎵 Introduction

A **vector space** $V$ over a **field** $\mathbb{R}$ is defined using:

- A **set of vectors** $V$
    
- A **field of scalars** $\mathbb{R}$
    
- Two operations:
    
    - **Vector addition** $+$
        
    - **Scalar multiplication** $\cdot$
        

In machine learning we mostly use vector spaces over **real numbers**.

---

# 1️⃣ Definition of a Vector Space

A set $V$ is a vector space over $\mathbb{R}$ if:

### Operation 1: Vector Addition

A binary operation  
$$ + : V \times V \to V $$

### Operation 2: Scalar Multiplication

A binary operation  
$$ \cdot : \mathbb{R} \times V \to V $$

---

# 2️⃣ Axioms of a Vector Space

For all scalars $a, b \in \mathbb{R}$ and vectors $u, v, w \in V$:

### **(1) $(V,+)$ is an Abelian group**

#### • Closure

$$ v + w \in V $$

#### • Commutativity

$$ v + w = w + v $$

#### • Associativity

$$ u + (v + w) = (u + v) + w $$

#### • Additive Identity

There exists $0 \in V$ such that  
$$ v + 0 = v $$

#### • Additive Inverse

For all $v \in V$,  
$$ v + (-v) = 0 $$

---

### **(2) Scalar closure**

$$ a \cdot v \in V $$

### **(3) Scalar distributes over vector addition**

$$ a(v + w) = av + aw $$

### **(4) Scalars distribute over scalar addition**

$$ (a + b)v = av + bv $$

### **(5) Associativity of scalar multiplication**

$$ a(bv) = (ab)v $$

### **(6) Unitary law**

$$ 1 \cdot v = v $$

If all six axioms hold → **$V$ is a vector space**.

---

# 3️⃣ Example: $\mathbb{R}^2$ is a Vector Space

Let  
$$ v = (a,b), \quad w = (c,d) $$

### ✔ Closure

$$ (a,b) + (c,d) = (a+c, , b+d) \in \mathbb{R}^2 $$

### ✔ Commutativity

$$ (a,b) + (c,d) = (c,d) + (a,b) $$

### ✔ Associativity

$$ (a,b) + ((c,d)+(e,f)) = ((a,b)+(c,d)) + (e,f) $$

### ✔ Additive Identity

$$ (0,0) + (a,b) = (a,b) $$

### ✔ Additive Inverse

$$ (a,b) + (-a,-b) = (0,0) $$

### ✔ Scalar multiplication closure

For $k \in \mathbb{R}$:  
$$ k(a,b) = (ka, kb) \in \mathbb{R}^2 $$

All remaining axioms follow similarly →  
**$\mathbb{R}^2$ is a vector space.**

Likewise:

- $\mathbb{R}^3$
    
- $\mathbb{R}^n$
    

are vector spaces.

---

# 4️⃣ More Vector Space Examples

### ✔ Matrices

All $m \times n$ matrices form a vector space under:

- Usual matrix addition
    
- Scalar multiplication
    

### ✔ Polynomials

- Set of all polynomials with real coefficients
    
- Set of all polynomials of degree ≤ $n$
    

### ✔ Convergent sequences

Also form a vector space.

---

# 5️⃣ Non-Examples (NOT Vector Spaces)

## ❌ 1. Polynomials of _exact_ degree $n$

Example: degree-3 polynomials.

Let:

- $p_1(x) = 2x^3 + 5x^2 + x + 7$
    
- $p_2(x) = -2x^3 + 3x^2 + 4x + 1$
    

Sum:  
$$  
p_1(x) + p_2(x)  
= 8x^2 + 5x + 8  
$$

Degree is 2 → **not in the set** → closure fails.

---

## ❌ 2. Modified vector addition in $\mathbb{R}^2$

Define:  
$$  
(x_1,y_1) \oplus (x_2,y_2)  
= (x_1 + x_2,; y_1 + 2y_2)  
$$

Reverse order:  
$$  
(x_2,y_2) \oplus (x_1,y_1)  
= (x_2 + x_1,; y_2 + 2y_1)  
$$

These are **not equal** unless $y_1 = y_2$ → commutativity fails → not a vector space.

**Lesson:**

> A set is not enough — the operations must satisfy all axioms.

---

# 6️⃣ Geometric Interpretation

## For $n = 1$

- Real line
    
- Vector from $0$ to $x$
    

Examples:

- $v = 5$ → arrow from 0 to 5
    
- $v = -2$ → arrow from 0 to −2
    

---

## For $n = 2$

Vector $(x,y)$ drawn as arrow from origin to $(x,y)$.

---

## For $n = 3$

Vector $(x,y,z)$ drawn from origin to $(x,y,z)$.

### Vector addition (parallelogram law)

If $v_1$ and $v_2$ are two vectors,  
$v_1 + v_2$ is the diagonal of the parallelogram.

---

## Linear Independence (geometric meaning)

Vectors $v_1, v_2, v_3$ in $\mathbb{R}^3$ are linearly independent if:

- $v_1$ and $v_2$ are not collinear
    
- $v_3$ does not lie in the plane of $v_1$ and $v_2$
    

---

# 7️⃣ Euclidean Space $\mathbb{R}^n$

Vector:  
$$  
x = (x_1, x_2, \dots, x_n)  
$$

Used to represent:

- Data points
    
- Feature vectors
    

ML commonly uses very large $n$ (100, 2000, etc.)

Visualization is only possible for $n \le 3$.

---
