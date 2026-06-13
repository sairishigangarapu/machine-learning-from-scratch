## Orthogonal Complement and Projection Mapping

*Essential Mathematics for ML — Structured Notes*

---

## 1. Orthogonal Vectors

### Motivation and Intuition
Imagine we have a dataset of 2D points $(x, y)$, and we want to draw a "line of best fit" through them using Linear Regression. In the vast majority of real-world datasets, a perfectly straight line cannot exactly hit every single scattered point. There is always some leftover error.

Geometrically, Linear Regression attempts to project the target data onto exactly the shadow of your model's capabilities. The "error" (the distance from the prediction line to the actual point) is minimized exclusively when that error vector points at a strict, perfect mathematically 90-degree angle away from the prediction line. This 90-degree relationship is known as **Orthogonality**.

### Formal Definition
In an inner product space, two vectors $v_i, v_j \in V$ are **orthogonal** if their inner product (dot product) is zero:

$$
\langle v_i, v_j \rangle = 0
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $\langle v_i, v_j \rangle$ | Inner product of vectors $v_i$ and $v_j$ | Zero value indicates perpendicularity |
| $0$ | Zero result | No alignment between the vectors |
| $v_i, v_j$ | Vectors in inner product space | Distinct basis directions |

If $\langle v_i, v_j \rangle = 0$, they are entirely independent of each other. Moving along axis $v_i$ gets you zero distance along axis $v_j$.

---

## 2. Orthogonal Complements

Let $W$ be a subspace (like the 2D plane our model predicts on). The **orthogonal complement** $W^\perp$ is the set of all vectors that are orthogonal to *every* vector in $W$.

$$
W^\perp = \{ v \in V \mid \langle v, w \rangle = 0, \forall w \in W \}
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $W^\perp$ | Orthogonal complement of subspace $W$ | Subspace of all vectors perpendicular to $W$ |
| $v \in V$ | Vector in ambient space $V$ | Candidate vector tested for orthogonality |
| $\langle v, w \rangle = 0$ | Inner product condition | Must hold for every $w$ in $W$ |
| $\forall w \in W$ | Universal quantifier | Ensures orthogonality to entire subspace |

**Why it matters:** In Linear Regression, if $W$ is our model's feature space, then $W^\perp$ is the exact space where our unexplainable "residual error" lives.

### Direct Sum Decomposition
Any vector in the entire universe $V$ can be cleanly sliced into two components: the part our model understands ($W$), and the orthogonal error ($W^\perp$).

$$
V = W \oplus W^\perp
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $V$ | Ambient vector space | The entire space being decomposed |
| $W$ | Subspace (e.g., model's feature space) | The "signal" component of vectors |
| $W^\perp$ | Orthogonal complement | The "noise" or residual component |
| $\oplus$ | Direct sum operator | Combines subspaces with trivial intersection |

$$
v = v_w + v_{w^\perp}
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $v$ | Original vector | Any point in ambient space |
| $v_w$ | Projection onto $W$ | Component captured by the model |
| $v_{w^\perp}$ | Component in $W^\perp$ | Residual error orthogonal to model |

---

## 3. Orthogonal Projections

To actively slice our data into those two components, we wield **Orthogonal Projections**.

Let $\{w_1, \dots, w_k\}$ be an orthogonal basis of $W$. The projection operator $P_W$ squashes any vector $v$ straight down onto $W$:

$$
P_W(v) = \sum_{i=1}^{k} \frac{\langle v, w_i \rangle}{\|w_i\|^2} w_i
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $P_W(v)$ | Orthogonal projection of $v$ onto $W$ | Closest point in $W$ to $v$ |
| $\langle v, w_i \rangle$ | Inner product with basis vector $w_i$ | Measures alignment with each basis direction |
| $\|w_i\|^2$ | Squared norm of basis vector | Normalizes the projection coefficient |
| $w_i$ | Orthogonal basis vectors of $W$ | Directions defining the subspace |
| $k$ | Dimension of subspace $W$ | Number of basis vectors |

```python
import numpy as np

v = np.array([0, 3, 10])
w1 = np.array([3, 0, 1])

# Projecting v strictly onto the 1D subspace spanned by w1
# Formula: (v dot w1 / w1 dot w1) * w1
projection = (np.dot(v, w1) / np.dot(w1, w1)) * w1

# Output is [3.0, 0.0, 1.0]. This is the "shadow" of v along w1.
```

### Example Calculation

Project $v = (0, 3, 10)$ onto the orthogonal plane spanned by $w_1 = (3, 0, 1)$ and $w_2 = (0, 1, 0)$.

1. **Calculate Term 1:** $\langle v, w_1 \rangle = 10$, $\|w_1\|^2 = 10 \implies \frac{10}{10} = 1$.
2. **Calculate Term 2:** $\langle v, w_2 \rangle = 3$, $\|w_2\|^2 = 1 \implies \frac{3}{1} = 3$.
3. **Compute Total Projection:**

$$
P_W(v) = 1(3, 0, 1) + 3(0, 1, 0) = (3, 3, 1)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $P_W(v)$ | Orthogonal projection of $v$ onto $W$ | The closest point in subspace $W$ to the original vector $v$ |
| $1, 3$ | Projection coefficients | Scalar weights computed as $\langle v, w_i \rangle / \|w_i\|^2$ |
| $(3, 0, 1)$ | Basis vector $w_1$ | First direction spanning the subspace $W$ |
| $(0, 1, 0)$ | Basis vector $w_2$ | Second direction spanning the subspace $W$ |
| $(3, 3, 1)$ | Resulting projected point | The shadow of $v$ onto the plane spanned by $w_1, w_2$ |

### The Residual Error
The residual vector is exactly $v - P_W(v) = (0, 3, 10) - (3, 3, 1) = (-3, 0, 9)$.
If we take the dot product of this residual with our plane vectors, the universe demands it be zero.
$\langle (-3, 0, 9), (3, 0, 1) \rangle = -9 + 0 + 9 = 0$. Perfect.

---

## 4. Projection Properties and Machine Learning Optimization

Orthogonal projections have distinct mathematical properties that algorithms routinely exploit:

1. **Idempotence ($P^2 = P$):** If you project a shadow onto the floor, predicting the projection of the shadow just gives you the shadow again.
2. **Best Approximation Theorem (The Core of Linear ML):**

$$
\|v - P_W(v)\| \le \|v - w\| \quad \forall w \in W
$$

| Term | Definition | Significance |
|------|------------|--------------|
| $\|v - P_W(v)\|$ | Distance from $v$ to its projection | Minimum possible error in subspace $W$ |
| $\|v - w\|$ | Distance from $v$ to arbitrary $w \in W$ | Error of any other candidate point |
| $\le$ | Less than or equal | Projection is at least as good as any alternative |
| $\forall w \in W$ | Universal quantifier | Holds for every point in the subspace |

This theorem guarantees that $P_W(v)$ is the absolute closest possible point to $v$ inside the subspace $W$. When an Ordinary Least Squares (OLS) algorithm runs, it doesn't arbitrarily guess a line. It executes exactly this orthogonal projection algebraically, guaranteeing it has mathematically achieved the lowest possible Mean Squared Error.

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 07: Norms](Lecture%2007%20Norms%20and%20Spaces.md) — Norms and inner products provide the distance and angle measures that define orthogonality
- **Next:** [Lecture 09: Eigenvalues](Lecture%2009%20Eigenvalues%20and%20Eigenvectors.md) — Eigenvectors of symmetric matrices are mutually orthogonal, connecting projection to eigendecomposition
- **Related:** [Lecture 27: Gram-Schmidt](Lecture%2027%20Gram%20Schmidt%20Process.md) — Algorithmic construction of orthogonal bases for projection
- **Related:** [Lecture 21: Least Squares](Lecture%2021%20Least%20Square%20Approximation%20and%20Minimum%20Norm%20Solution.md) — Direct application of orthogonal projection to solve overdetermined systems