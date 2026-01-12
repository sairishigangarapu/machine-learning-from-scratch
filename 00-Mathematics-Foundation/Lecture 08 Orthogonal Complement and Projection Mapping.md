
## 1. Orthogonal Vectors

In an inner product space $(V, \langle \cdot, \cdot \rangle)$, two vectors $v_i, v_j \in V$ are said to be **orthogonal** if their inner product is zero:

$$\langle v_i, v_j \rangle = 0$$

### Generalisation of Perpendicularity

- In Euclidean space ($\mathbb{R}^n$) with the standard dot product, orthogonality is equivalent to the geometric concept of being perpendicular.
    
- If the dot product of two vectors is zero, they are perpendicular.
    

Example:

Let $V = \mathbb{R}^3$ with the standard dot product.

Vectors $v_1 = (1, 0, 0)$ and $v_2 = (0, 1, 0)$ are orthogonal because:

$$\langle v_1, v_2 \rangle = 1\cdot0 + 0\cdot1 + 0\cdot0 = 0$$

Geometrically: $v_1$ lies on the x-axis and $v_2$ lies on the y-axis, which are perpendicular.

---

## 2. Orthogonal Complements

Let $W$ be a subset of an inner product space $V$. The **orthogonal complement** of $W$, denoted by $W^\perp$, is the set of all vectors in $V$ that are orthogonal to _every_ element in $W$.

### Definition

$$W^\perp = \{ v \in V \mid \langle v, w \rangle = 0, \forall w \in W \}$$

### Properties

1. **Subspace Property:** Regardless of whether $W$ is a subspace, $W^\perp$ is _always_ a subspace of $V$.
    
2. Direct Sum Decomposition: If $W$ is a subspace of $V$, then $V$ can be written as the direct sum of $W$ and $W^\perp$:
    
    $$V = W \oplus W^\perp$$
    
    This means every vector $v \in V$ can be uniquely decomposed as:
    
    $$v = v_w + v_{w^\perp}$$
    
    where $v_w \in W$ and $v_{w^\perp} \in W^\perp$.
    

Example:

Let $V = \mathbb{R}^3$.

Let $W = \text{span}\{(1, 0, 0), (0, 1, 0)\}$.

- $W$ represents the xy-plane.
    
- The orthogonal complement $W^\perp$ is the subspace spanned by $(0, 0, 1)$ (the z-axis).
    
- Any vector $(2, 3, 5) \in \mathbb{R}^3$ can be decomposed:
    
    $$(2, 3, 5) = \underbrace{(2, 3, 0)}_{\in W} + \underbrace{(0, 0, 5)}_{\in W^\perp}$$
    

---

## 3. Orthogonal Projections

Let $V$ be an inner product space and $W$ be a subspace of $V$. Let $\{w_1, w_2, \dots, w_k\}$ be an **orthogonal basis** of $W$ (vectors are mutually orthogonal).

### Definition

The orthogonal projection $P_W: V \to W$ is a linear transformation defined for any $v \in V$ as:

$$P_W(v) = \sum_{i=1}^{k} \frac{\langle v, w_i \rangle}{\|w_i\|^2} w_i$$

### Example Calculation

Let $W = \text{span}\{w_1, w_2\}$ where $w_1 = (3, 0, 1)$ and $w_2 = (0, 1, 0)$ in $\mathbb{R}^3$.

Find the projection of $v = (0, 3, 10)$ onto $W$.

1. **Check Orthogonality of Basis:** $\langle w_1, w_2 \rangle = 0$. (Orthogonal basis verified).
    
2. **Calculate Terms:**
    
    - $\langle v, w_1 \rangle = 0(3) + 3(0) + 10(1) = 10$
        
    - $\|w_1\|^2 = 3^2 + 0^2 + 1^2 = 10$
        
    - $\langle v, w_2 \rangle = 0(0) + 3(1) + 10(0) = 3$
        
    - $\|w_2\|^2 = 0^2 + 1^2 + 0^2 = 1$
        
3. Compute Projection:
    
    $$P_W(v) = \frac{10}{10} (3, 0, 1) + \frac{3}{1} (0, 1, 0)$$
    
    $$P_W(v) = (3, 0, 1) + (0, 3, 0) = (3, 3, 1)$$
    
4. Verification:
    
    The residual vector is $v - P_W(v) = (0, 3, 10) - (3, 3, 1) = (-3, 0, 9)$.
    
    Check orthogonality to $w_1$: $\langle (-3, 0, 9), (3, 0, 1) \rangle = -9 + 0 + 9 = 0$.
    

### Properties of Orthogonal Projections

1. **Orthogonality of Residual:** For any $v \in V$, the vector $v - P_W(v)$ is orthogonal to $W$ (i.e., $v - P_W(v) \in W^\perp$).
    
2. **Idempotence on Subspace:** If $w \in W$, then $P_W(w) = w$.
    
3. **Idempotence of Operator:** $P^2 = P$ (applying projection twice changes nothing).
    
4. **Norm Reduction:** $\|P_W(v)\| \le \|v\|$ for all $v \in V$.
    
5. Best Approximation:
    
    $$\|v - P_W(v)\| \le \|v - w\| \quad \forall w \in W$$
    
    $P_W(v)$ is the vector in $W$ that is closest to $v$. It minimizes the distance between $v$ and the subspace $W$.
    

---

## 4. Projection Transformations

A linear transformation $P: V \to V$ is called a Projection if it satisfies the idempotent property:

$$P^2 = P$$

- **Orthogonal Projections** are a specific type of projection where the range and null space are orthogonal complements.
    
- The **Identity Transformation** ($I$) is a trivial projection ($I^2 = I$).
    

### Application: Finding the Closest Point

The "Best Approximation" property is critical in machine learning (e.g., finding the nearest vector of a specific class).

**Problem:** Find the closest point to $v = (2, 4, 0, -2)$ in the subspace $W = \text{span}\{(1, 1, 0, 0), (0, 0, 1, 1)\}$.

Solution:

Let $w_1 = (1, 1, 0, 0)$ and $w_2 = (0, 0, 1, 1)$. These are orthogonal.

The closest point is the orthogonal projection $\hat{w} = P_W(v)$.

$$\hat{w} = \frac{\langle v, w_1 \rangle}{\|w_1\|^2} w_1 + \frac{\langle v, w_2 \rangle}{\|w_2\|^2} w_2$$

1. $\langle v, w_1 \rangle = 2(1) + 4(1) = 6$; $\|w_1\|^2 = 2$.
    
2. $\langle v, w_2 \rangle = 0(1) + (-2)(1) = -2$; $\|w_2\|^2 = 2$.
    
3. $\hat{w} = \frac{6}{2}(1, 1, 0, 0) + \frac{-2}{2}(0, 0, 1, 1)$
    
4. $\hat{w} = 3(1, 1, 0, 0) - 1(0, 0, 1, 1) = (3, 3, -1, -1)$
    

Result: The closest point in $W$ to $v$ is $(3, 3, -1, -1)$.