## Introduction to Vector Spaces

*Essential Mathematics for ML — Structured Notes*

---

## 1. Definition of a Vector Space

A **vector space** $V$ over a **field** $\mathbb{R}$ is a mathematical structure consisting of a set of vectors, a field of scalars (typically real numbers in ML), and two fundamental operations that satisfy specific properties.

### Primary Operations

1. **Vector Addition ($+$):** A binary operation that maps two vectors to a third vector within the same space: $+ : V \times V \to V$.
2. **Scalar Multiplication ($\cdot$):** A binary operation that scales a vector by a real number: $\cdot : \mathbb{R} \times V \to V$.

---

## 2. Axioms of a Vector Space

For a set $V$ to be considered a vector space, it must satisfy the following axioms for all scalars $a, b \in \mathbb{R}$ and vectors $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$:

### (A) Additive Properties (Abelian Group)

* **Closure:** $\mathbf{v} + \mathbf{w} \in V$.
* **Commutativity:** $\mathbf{v} + \mathbf{w} = \mathbf{w} + \mathbf{v}$.
* **Associativity:** $\mathbf{u} + (\mathbf{v} + \mathbf{w}) = (\mathbf{u} + \mathbf{v}) + \mathbf{w}$.
* **Additive Identity:** There exists a zero vector $\mathbf{0} \in V$ such that $\mathbf{v} + \mathbf{0} = \mathbf{v}$.
* **Additive Inverse:** For every $\mathbf{v} \in V$, there exists $-\mathbf{v}$ such that $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$.

### (B) Multiplicative and Distributive Properties

* **Scalar Closure:** $a \cdot \mathbf{v} \in V$.
* **Distributivity (Vector):** $a(\mathbf{v} + \mathbf{w}) = a\mathbf{v} + a\mathbf{w}$.
* **Distributivity (Scalar):** $(a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$.
* **Associativity of Scalars:** $a(b\mathbf{v}) = (ab)\mathbf{v}$.
* **Unitary Law:** $1 \cdot \mathbf{v} = \mathbf{v}$.

---

## 3. Valid Vector Space Examples

### Euclidean Spaces ($\mathbb{R}^n$)

$\mathbb{R}^2, \mathbb{R}^3$, and general $\mathbb{R}^n$ are the most common vector spaces in Machine Learning. In $\mathbb{R}^2$, given $\mathbf{v} = (a,b)$ and $\mathbf{w} = (c,d)$:

* Addition: $(a+c, b+d) \in \mathbb{R}^2$.
* Scalar Multiplication: $k(a,b) = (ka, kb) \in \mathbb{R}^2$.

### Other Technical Examples

* **Matrices:** The set of all $m \times n$ matrices forms a vector space under standard matrix addition and scalar multiplication.
* **Polynomials:** The set of all polynomials of degree $\le n$ is a vector space.
* **Convergent Sequences:** These form a vector space as the sum of two convergent sequences also converges.

---

## 4. Non-Examples (Counter-Cases)

Understanding where axioms fail is critical for identifying invalid structures:

* **Exact Degree Polynomials:** The set of polynomials of *exactly* degree $n$ is not a vector space. For example, adding $p_1(x) = 2x^3$ and $p_2(x) = -2x^3 + x^2$ results in $x^2$. The degree changes from 3 to 2, violating **closure**.
* **Modified Operations:** If we redefine addition as $(x_1, y_1) \oplus (x_2, y_2) = (x_1 + x_2, y_1 + 2y_2)$, the operation is no longer **commutative**, thus failing the Abelian group requirement.

---

## 5. Geometric Interpretation

### Visualizing Vectors

* **$n=1$:** A point or arrow on the real number line.
* **$n=2, 3$:** An arrow originating from the origin $(0,0)$ to a coordinate point.

### Parallelogram Law of Addition

Geometrically, the sum of two vectors $\mathbf{v}_1 + \mathbf{v}_2$ is represented by the diagonal of the parallelogram formed by those vectors as adjacent sides.

### Linear Independence

In $\mathbb{R}^3$, vectors are linearly independent if:

1. They are not collinear (do not lie on the same line).
2. The third vector does not lie in the plane created by the first two.

---

## 6. Applications in $\mathbb{R}^n$

In Machine Learning, $\mathbb{R}^n$ is the standard environment for:

* **Data Points:** Each sample is a point in $n$-dimensional space.
* **Feature Vectors:** Numerical representations of categorical or continuous data.

While we can only visualize up to $n=3$, ML models routinely operate in high-dimensional spaces (e.g., $n=1024$ for word embeddings). The algebraic axioms defined above ensure that calculations remain consistent regardless of the number of dimensions.

---
