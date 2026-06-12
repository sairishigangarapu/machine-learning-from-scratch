## Definiteness of Matrices

*Essential Mathematics for ML — Structured Notes*

---

## 1. Definition of Definiteness

### Motivation and Intuition
When we train a neural network, we analyze the curvature of the loss landscape using the **Hessian matrix** (the matrix of second derivatives). If the Hessian is positive definite everywhere, the loss surface is a perfect bowl — gradient descent converges cleanly. If the Hessian has mixed signs, the surface has saddle points where the optimizer gets stuck. **Matrix definiteness** is the mathematical language that classifies these behaviors.

### Formal Definition
A symmetric matrix $H \in \mathbb{R}^{n \times n}$ is classified as:

| Type | Condition |
|:---|:---|
| **Positive Definite (PD)** | $\mathbf{x}^T H \mathbf{x} > 0 \quad \forall \mathbf{x} \neq \mathbf{0}$ |
| **Positive Semi-Definite (PSD)** | $\mathbf{x}^T H \mathbf{x} \ge 0 \quad \forall \mathbf{x}$ |
| **Negative Definite (ND)** | $\mathbf{x}^T H \mathbf{x} < 0 \quad \forall \mathbf{x} \neq \mathbf{0}$ |
| **Negative Semi-Definite (NSD)** | $\mathbf{x}^T H \mathbf{x} \le 0 \quad \forall \mathbf{x}$ |
| **Indefinite** | Takes both positive and negative values |

### Remark
$H$ is negative (semi-)definite if and only if $-H$ is positive (semi-)definite. We only need to study the positive cases.

---

## 2. Geometric Interpretation: The "Bowl" Test

The quadratic form $\mathbf{x}^T H \mathbf{x}$ defines a multivariate parabola.

* **PD:** The parabola curves **upward in every direction** — a perfect bowl. Single, unique minimum.
* **PSD:** The parabola is flat in some directions — a bowl with a flat bottom. Minimum is a set, not a point.
* **ND:** The parabola curves **downward in every direction** — an upside-down bowl. Single, unique maximum.
* **Indefinite:** The parabola curves up in some directions, down in others — a **saddle point**. Neither a minimum nor a maximum.

```python
import numpy as np

def classify_definiteness(H):
    """Classify a symmetric matrix by its definiteness."""
    eigenvalues = np.linalg.eigvalsh(H)
    
    if np.all(eigenvalues > 0):
        return "Positive Definite"
    elif np.all(eigenvalues >= 0):
        return "Positive Semi-Definite"
    elif np.all(eigenvalues < 0):
        return "Negative Definite"
    elif np.all(eigenvalues <= 0):
        return "Negative Semi-Definite"
    else:
        return "Indefinite"

# PD: bowl
H1 = np.array([[3, 1], [1, 3]])
print(classify_definiteness(H1))  # Positive Definite

# Indefinite: saddle
H2 = np.array([[1, 0], [0, -1]])
print(classify_definiteness(H2))  # Indefinite
```

---

## 3. Testing Definiteness

### Method 1: Eigenvalues

| Eigenvalues | Classification |
|:---|:---|
| All $\lambda_i > 0$ | Positive Definite |
| All $\lambda_i \ge 0$ (at least one $= 0$) | Positive Semi-Definite |
| All $\lambda_i < 0$ | Negative Definite |
| Mixed signs | Indefinite |

```python
import numpy as np

H = np.array([[2, -1, 0],
              [-1, 2, -1],
              [0, -1, 2]])

eigenvalues = np.linalg.eigvalsh(H)
print(f"Eigenvalues: {eigenvalues}")  # All positive
print(f"Classification: Positive Definite")
```

### Method 2: Leading Principal Minors (Sylvester's Criterion)

For a symmetric matrix $H$:

* **PD:** All leading principal minors have positive determinant.
* **ND:** Leading principal minors alternate in sign (first $< 0$, second $> 0$, ...).

```python
import numpy as np

def leading_principal_minors(H):
    """Compute all leading principal minors."""
    n = H.shape[0]
    minors = []
    for k in range(1, n+1):
        submatrix = H[:k, :k]
        minors.append(np.linalg.det(submatrix))
    return minors

H = np.array([[2, -1], [-1, 2]])
minors = leading_principal_minors(H)
print(f"Minors: {minors}")  # [2, 3] — all positive → PD
```

### Method 3: Cholesky Decomposition

A matrix is PD if and only if it has a Cholesky decomposition $H = LL^T$ where $L$ is lower triangular with positive diagonal entries.

```python
import numpy as np

H = np.array([[4, 2], [2, 3]])

try:
    L = np.linalg.cholesky(H)
    print(f"Cholesky factor:\n{L}")
    print("Matrix is Positive Definite")
except np.linalg.LinAlgError:
    print("Matrix is NOT Positive Definite")
```

---

## 4. Worked Examples

### Example 1: Positive Definite

$$
H = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}
$$

**Eigenvalues:** $\lambda_1 = 4, \lambda_2 = 2$ — both positive.

**Direct check:** $\mathbf{x}^T H \mathbf{x} = 3x_1^2 + 2x_1 x_2 + 3x_2^2 = 2x_1^2 + (x_1 + x_2)^2 + 2x_2^2 > 0$ for $\mathbf{x} \neq \mathbf{0}$.

**Geometric meaning:** This defines a bowl-shaped loss surface. Gradient descent converges to the unique minimum.

### Example 2: Indefinite

$$
H = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}
$$

**Eigenvalues:** $\lambda_1 = 1, \lambda_2 = -1$ — mixed signs.

**Direct check:** $\mathbf{x}^T H \mathbf{x} = x_1^2 - x_2^2$. At $(1, 0)$ this is $+1$, at $(0, 1)$ this is $-1$.

**Geometric meaning:** The origin is a saddle point — the function curves up along $x_1$ and down along $x_2$. Gradient descent oscillates without converging.

### Example 3: Positive Semi-Definite (Singular)

$$
H = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}
$$

**Eigenvalues:** $\lambda_1 = 2, \lambda_2 = 0$.

**Geometric meaning:** The function is flat along the direction $(1, -1)$. The minimum is not unique — it's an entire line of minima.

---

## 5. Definiteness and the Loss Landscape

### The Hessian of a Loss Function

For a loss function $\mathcal{L}(\boldsymbol{\theta})$ with parameters $\boldsymbol{\theta} \in \mathbb{R}^d$:

* **At a local minimum:** $H \succeq 0$ (positive semi-definite).
* **At a local maximum:** $H \preceq 0$ (negative semi-definite).
* **At a saddle point:** $H$ is indefinite.

**Condition number** $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$ measures how "narrow" the loss bowl is:

* $\kappa \approx 1$: Nearly spherical bowl → gradient descent converges fast.
* $\kappa \gg 1$: Elongated bowl → gradient descent oscillates and converges slowly.

```python
import numpy as np

def condition_number(H):
    eigenvalues = np.linalg.eigvalsh(H)
    return max(eigenvalues) / min(eigenvalues[eigenvalues > 0])

# Well-conditioned
H_good = np.array([[1, 0], [0, 1.5]])
print(f"kappa (good): {condition_number(H_good):.2f}")  # 1.5

# Ill-conditioned
H_bad = np.array([[1, 0], [0, 100]])
print(f"kappa (bad): {condition_number(H_bad):.2f}")  # 100
```

---

## 6. Connection to Convexity (Lecture 38)

A differentiable function $f$ is convex if and only if its Hessian is positive semi-definite everywhere:

$$
f \text{ convex} \iff \nabla^2 f(\mathbf{x}) \succeq 0 \quad \forall \mathbf{x}
$$

| Hessian Properties | Loss Surface | Optimization Behavior |
|:---|:---|:---|
| PD everywhere | Perfect bowl | GD converges to unique global minimum |
| PSD everywhere (some $\lambda = 0$) | Bowl with flat bottom | Multiple minima (a subspace of minima) |
| Indefinite at some points | Saddle points exist | GD may get stuck or oscillate |
| Negative eigenvalues | Local maxima exist | GD may diverge |

**Deep Learning Failure Mode:** In deep networks with thousands of parameters, the Hessian is almost always indefinite at random initialization — saddle points dominate the loss landscape. Second-order methods (like Newton's method) would move *toward* saddle points if the Hessian is indefinite, which is catastrophic. This is why first-order methods (SGD, Adam) are preferred — they only use gradient information and are immune to saddle point issues via noise.

---

## 7. Special Matrix Classes and Definiteness

| Matrix Type | Definiteness |
|:---|:---|
| **Covariance matrix** $\Sigma$ | Always PSD (eigenvalues = variances $\ge 0$) |
| **Gram matrix** $X^TX$ | Always PSD |
| **Hessian of convex function** | Always PSD |
| **$A^TA$** (any $A$) | Always PSD |
| **$AA^T$** (any $A$) | Always PSD |
| **Correlation matrix** | Always PSD |

```python
import numpy as np

# Covariance matrix is always PSD
X = np.random.randn(100, 5)
Sigma = np.cov(X, rowvar=False)
eigenvalues = np.linalg.eigvalsh(Sigma)
print(f"All eigenvalues >= 0: {np.all(eigenvalues >= -1e-10)}")  # True
```

---

## 8. Summary

| Type | $\mathbf{x}^T H \mathbf{x}$ | Eigenvalues | Loss Surface |
|:---|:---|:---|:---|
| **PD** | $> 0$ for $\mathbf{x} \neq 0$ | All $> 0$ | Bowl (unique minimum) |
| **PSD** | $\ge 0$ | All $\ge 0$ | Flat bowl (multiple minima) |
| **ND** | $< 0$ for $\mathbf{x} \neq 0$ | All $< 0$ | Inverted bowl |
| **Indefinite** | Mixed | Mixed signs | Saddle point |

> **Check your intuition:** If the Hessian at a critical point has eigenvalues $\{3, 0, -2\}$, is it a minimum, maximum, or saddle point? *(Answer: Saddle point. The positive eigenvalue (3) means the function curves up in one direction, the negative eigenvalue (-2) means it curves down in another, and the zero eigenvalue means it's flat in a third direction. It is neither a min nor a max.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 38: Convex Functions](Lecture%2038%20Convex%20Functions.md) — introduces the second-order condition for convexity, which requires positive semi-definiteness of the Hessian
- **Next:** [Lecture 40: Optimality Conditions](Lecture%2040%20Optimality%20Conditions.md) — applies Hessian definiteness to classify critical points as minima, maxima, or saddle points
- **Related:** [Lecture 10: Special Matrices and Properties](Lecture%2010%20Special%20Matrices%20and%20Properties.md) — provides the matrix algebra foundation (eigenvalues, spectral properties) for understanding definiteness
- **Related:** [Lecture 38: Convex Functions](Lecture%2038%20Convex%20Functions.md) — convexity is characterized by the Hessian being positive semi-definite everywhere
- **Related:** [Lecture 41: Unconstrained Optimization](Lecture%2041%20Unconstrained%20Optimization.md) — uses Hessian definiteness at critical points to determine if they are local minima
