## Convex Sets

*Essential Mathematics for ML — Structured Notes*

---

## 1. Definition of a Convex Set

### Motivation and Intuition
Why do we care about convexity? Because convexity is the mathematical property that guarantees optimization problems have a **single, global optimum**. When a machine learning loss function is convex, gradient descent is mathematically guaranteed to find the best possible solution — no local traps, no saddle points, no ambiguity. Convex sets are the geometric foundation of convex functions.

### Formal Definition
A set $S \subseteq \mathbb{R}^n$ is **convex** if for every pair of points $\mathbf{x}_1, \mathbf{x}_2 \in S$ and every $\lambda \in [0, 1]$:

$$
\lambda \mathbf{x}_1 + (1 - \lambda)\mathbf{x}_2 \in S
$$

**Geometric meaning:** The straight line segment connecting any two points in $S$ lies entirely inside $S$. No part of the segment "dips outside" the set.

```python
import numpy as np

# Check convexity: for any two points, the line segment must stay in S
def is_convex(points, num_tests=1000):
    """Monte Carlo check for convexity of a 2D set."""
    for _ in range(num_tests):
        i, j = np.random.choice(len(points), 2, replace=False)
        p1, p2 = points[i], points[j]
        lam = np.random.uniform(0, 1)
        segment_point = lam * p1 + (1 - lam) * p2
        # Check if segment_point is in S (this depends on your set)
        # For a disk: x^2 + y^2 <= 1
        if segment_point[0]**2 + segment_point[1]**2 > 1:
            return False
    return True
```

---

## 2. Visual Examples

### Convex Sets

* **Disk:** $\{(x,y) : x^2 + y^2 \le 1\}$ — any line segment between two interior points stays inside.
* **Half-space:** $\{\mathbf{x} : \mathbf{a}^T\mathbf{x} \le b\}$ — always convex (it's a linear inequality).
* **Entire $\mathbb{R}^n$** — trivially convex.
* **Intersection of convex sets** — always convex (a powerful property).

### Non-Convex Sets

* **Annulus (ring):** $\{(x,y) : 1 \le x^2 + y^2 \le 4\}$ — a line segment crossing the hole exits the set.
* **Union of two disjoint disks** — a line segment connecting a point in one disk to a point in the other passes through empty space.
* **Any set with a "dent" or "cave"** — the line segment across the dent exits the set.

---

## 3. Properties of Convex Sets

### Property 1: Intersection
The intersection of any collection of convex sets is convex.

$$
S_1 \cap S_2 \cap \dots \cap S_k \text{ is convex if each } S_i \text{ is convex.}
$$

**ML Connection:** Feasible regions defined by multiple linear constraints (like $a_i^T\mathbf{x} \le b_i$ for $i = 1, \dots, m$) are intersections of half-spaces — always convex. This is why **linear programming** (constrained optimization with linear objective and constraints) always has a convex feasible region.

### Property 2: Union (NOT guaranteed)
The union of convex sets is generally **not** convex.

### Property 3: Affine Transformation
If $S$ is convex and $f(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$ is an affine mapping, then $f(S)$ is also convex.

**ML Connection:** A neural network layer $W\mathbf{x} + \mathbf{b}$ is an affine transformation. If the input set of representations is convex, the pre-activation values remain in a convex set (before the non-linearity).

---

## 4. Key Convex Sets in ML

### Hyperplane
$$
H = \{\mathbf{x} \in \mathbb{R}^n : \mathbf{a}^T\mathbf{x} = b\}
$$

A flat $(n-1)$-dimensional subspace. It divides $\mathbb{R}^n$ into two half-spaces. **Support Vector Machines** find the optimal hyperplane that separates two classes with maximum margin.

### Half-Space
$$
H^+ = \{\mathbf{x} \in \mathbb{R}^n : \mathbf{a}^T\mathbf{x} \le b\}
$$

Always convex. The decision boundary of a linear classifier is a half-space.

### Polytope
The intersection of finitely many half-spaces:

$$
P = \{\mathbf{x} : A\mathbf{x} \le \mathbf{b}\}
$$

Always convex. Used in constrained optimization problems.

### Ellipsoid
$$
E = \{\mathbf{x} : (\mathbf{x} - \mathbf{c})^T Q^{-1} (\mathbf{x} - \mathbf{c}) \le 1\}
$$

where $Q$ is positive definite. Always convex. **Covariance matrices** define ellipsoidal confidence regions in multivariate statistics.

```python
import numpy as np

# Ellipsoid defined by positive definite matrix Q
Q = np.array([[2, 0.5],
              [0.5, 1]])

# Check if a point is inside the ellipsoid
def in_ellipsoid(x, c, Q):
    diff = x - c
    return diff @ np.linalg.inv(Q) @ diff <= 1

c = np.array([0, 0])
print(in_ellipsoid(np.array([1, 1]), c, Q))   # True
print(in_ellipsoid(np.array([5, 5]), c, Q))   # False
```

---

## 5. Convex Hull

The **convex hull** of a set $S$ is the smallest convex set containing $S$. It is the set of all convex combinations:

$$
\text{conv}(S) = \left\{ \sum_{i=1}^{k} \lambda_i \mathbf{x}_i : \mathbf{x}_i \in S, \; \lambda_i \ge 0, \; \sum \lambda_i = 1 \right\}
$$

**Geometric intuition:** Stretch a rubber band around all points in $S$ — the region inside the rubber band is the convex hull.

**ML Connection:** The convex hull of training data points from two classes defines the region where a linear classifier can potentially separate them. If the convex hulls of the two classes overlap, no linear separator exists.

```python
from scipy.spatial import ConvexHull
import numpy as np

# Generate random points
points = np.random.randn(20, 2)

# Compute convex hull
hull = ConvexHull(points)
print(f"Vertices: {hull.vertices}")
print(f"Area: {hull.volume}")  # In 2D, volume = area
```

---

## 6. Cones

A set $C$ is a **cone** if for every $\mathbf{x} \in C$ and $\lambda \ge 0$:

$$
\lambda \mathbf{x} \in C
$$

A cone is **convex** if it also satisfies: $\mathbf{x}_1, \mathbf{x}_2 \in C \implies \mathbf{x}_1 + \mathbf{x}_2 \in C$.

**ML Connection:** The set of positive semi-definite matrices forms a convex cone. This is critical for understanding the geometry of covariance matrices and the Hessian in optimization.

---

## 7. Why Convexity Matters in ML

| Concept | Convexity Role |
|:---|:---|
| **Loss Landscape** | Convex loss → single global minimum → gradient descent converges |
| **SVM** | Quadratic programming over a convex feasible region |
| **Linear Regression** | MSE loss is convex → Normal Equation gives the unique global optimum |
| **Logistic Regression** | Log-loss is convex → guaranteed convergence |
| **Non-convex losses** | Neural networks → multiple local minima → no convergence guarantee |

> **Check your intuition:** Is the set $\{(x, y) : y \ge x^2\}$ convex? *(Answer: Yes. For any two points above the parabola, the line segment connecting them stays above the parabola. This is the epigraph of a convex function, and epigraphs of convex functions are always convex sets.)*
