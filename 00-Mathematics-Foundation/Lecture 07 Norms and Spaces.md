## 1. Metric Spaces

A **Metric Space** is a generalisation of the notion of distance from Euclidean space (e.g., distance in $\mathbb{R}^2$ or $\mathbb{R}^3$).

### Definition

A metric on a set $S$ is a function $d: S \times S \to \mathbb{R}$ satisfying the following properties for all $x, y, z \in S$:

1. Non-negativity:
    
    $$d(x, y) \ge 0$$
    
    (and $d(x, y) = 0 \iff x = y$)
    
2. Symmetry:
    
    $$d(x, y) = d(y, x)$$
    
3. Triangle Inequality:
    
    $$d(x, z) \le d(x, y) + d(y, z)$$
    

The set $S$ together with the metric $d$ is called a **Metric Space**, denoted as $(S, d)$.

Example:

For $S \subseteq \mathbb{R}$, define the metric as the absolute difference:

$$d(x, y) = |x - y|$$

---

## 2. Normed Spaces

A **Norm** is a generalization of the notion of **length** for vectors.

### Definition

A norm on a real Vector Space $V$ is a function $\|\cdot\|: V \to \mathbb{R}$ satisfying the following properties for all $x, y \in V$ and scalar $\alpha \in \mathbb{R}$:

1. Non-negativity:
    
    $$\|x\| \ge 0$$
    
    (and $\|x\| = 0 \iff x = \mathbf{0}$)
    
2. Scalar Multiplication:
    
    $$\|\alpha x\| = |\alpha| \cdot \|x\|$$
    
3. Triangle Inequality:
    
    $$\|x + y\| \le \|x\| + \|y\|$$
    

The vector space $V$ together with the norm is called a **Normed Space** (or Normed Linear Space).

### Relation between Metric and Normed Spaces

- **Every Normed Space is a Metric Space.** We can define the metric induced by the norm as $d(x, y) = \|x - y\|$.
    
- **The converse is NOT true.** Every Metric Space is not necessarily a Normed Space.
    

Counter-Example:

Let $X = \{0, 1\} \subset \mathbb{R}$. Define the Discrete Metric:

$$d(x, y) = \begin{cases} 1 & \text{if } x \neq y \\ 0 & \text{otherwise} \end{cases}$$

$(X, d)$ is a metric space but cannot be a normed space because it does not satisfy the properties of a norm (specifically regarding vector space structure and scalar multiplication).

### Common Norms on $\mathbb{R}^n$

For a vector $x = (x_1, x_2, \dots, x_n) \in \mathbb{R}^n$:

- L1 Norm (Manhattan Norm):
    
    $$\|x\|_1 = \sum_{i=1}^{n} |x_i|$$
    
- L2 Norm (Euclidean Norm):
    
    $$\|x\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}$$
    
- $p$-Norm:
    
    $$\|x\|_p = \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p}$$
    
- $L_\infty$ Norm (Max Norm):
    
    $$\|x\|_\infty = \max_{i} |x_i|$$
    

Example Calculation:

For vector $x = (1, 0, -2) \in \mathbb{R}^3$:

- $\|x\|_1 = |1| + |0| + |-2| = 3$
    
- $\|x\|_2 = \sqrt{1^2 + 0^2 + (-2)^2} = \sqrt{5}$
    
- $\|x\|_\infty = \max(1, 0, 2) = 2$
    

### Geometric Visualization

Geometrically, the "unit circles" (where $\|x\| = 1$) for these norms look different in 2D:

- **L2 Norm:** A standard circle.
    
- **L1 Norm:** A square rotated by 45 degrees (diamond shape).
    
- **$L_\infty$ Norm:** A standard square.

![[Pasted image 20260112231028.png]]

---

## 3. Convexity

All norms are **convex functions**.

### Convex Function

A function $f: S \to \mathbb{R}$ (where $S$ is a convex subset of $\mathbb{R}^n$) is convex if for all $x_1, x_2 \in S$ and $\lambda \in [0, 1]$:

$$f(\lambda x_1 + (1-\lambda)x_2) \le \lambda f(x_1) + (1-\lambda)f(x_2)$$

Geometrically: The chord joining any two points on the function graph lies above or on the graph.

### Convex Set

A set $S$ is convex if the line segment joining any two points in the set lies entirely within the set.

$$x_1, x_2 \in S \implies \lambda x_1 + (1-\lambda)x_2 \in S \quad \forall \lambda \in [0, 1]$$

- **Rectangles and Circular balls** are convex sets.
    
- **Kidney-bean shapes** are typically not convex sets (a line between two internal points may pass outside).
    
- **Geometric Norm Definition:** In real finite-dimensional vector spaces, any **symmetric, compact, convex region centered at the origin** defines a norm.
    

---

## 4. Inner Product Spaces

Inner product spaces generalize the dot product and are crucial for analyzing classifiers in machine learning.

### Definition

An inner product on a real Vector Space $V$ is a function $\langle \cdot, \cdot \rangle: V \times V \to \mathbb{R}$ satisfying:

1. Non-negativity:
    
    $$\langle x, x \rangle \ge 0 \quad (\text{and } \langle x, x \rangle = 0 \iff x = 0)$$
    
2. Linearity (in the first argument):
    
    $$\langle \alpha x, y \rangle = \alpha \langle x, y \rangle$$
    
    $$\langle x + y, z \rangle = \langle x, z \rangle + \langle y, z \rangle$$
    
3. Symmetry:
    
    $$\langle x, y \rangle = \langle y, x \rangle$$
    

A vector space equipped with an inner product is called an **Inner Product Space**.

### Examples

1. Standard Dot Product (in $\mathbb{R}^n$):
    
    $$\langle x, y \rangle = \sum_{i=1}^{n} x_i y_i$$
    
2. Weighted Inner Product (in $\mathbb{R}^2$):
    
    For $u = (u_1, u_2)$ and $v = (v_1, v_2)$, a valid inner product can be defined as:
    
    $$\langle u, v \rangle = 2u_1v_1 - u_1v_2 - v_1u_2 + u_2v_2$$
    

### Angle Between Vectors

The angle $\theta$ between vectors $x$ and $y$ is defined via the inner product:

$$\langle x, y \rangle = \|x\| \|y\| \cos \theta$$

---

## 5. The $L_0$ "Norm"

Strictly speaking, $L_0$ is **not a norm**, but it is widely used in compressed sensing and sparse solutions.

### Definition

The $L_0$ norm of a vector $x$ is the number of non-zero elements in $x$.

$$\|x\|_0 = |\{i : x_i \neq 0\}|$$

Why is it not a norm?

It fails the scalar multiplication property ($\|\alpha x\| = |\alpha| \|x\|$) for $\alpha \neq \pm 1$.

Example: If $x$ has 4 non-zero elements, $\|x\|_0 = 4$. If we scale $x$ by $\alpha=2$, the resulting vector $2x$ still has 4 non-zero elements ($2x \neq 2 \times 4$).