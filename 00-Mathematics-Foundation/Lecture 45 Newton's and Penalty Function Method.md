## Newton's Method

*Essential Mathematics for ML — Structured Notes*

---

## 1. Newton-Raphson: Finding Roots

### Motivation and Intuition
Gradient descent is a "first-order" method — it uses only the gradient (first derivative) to decide where to move. But the gradient only tells you the slope, not the curvature. A function might look steep but actually be approaching a minimum rapidly, or look flat but actually be in a wide, shallow valley. **Newton's method** uses the second derivative (curvature) to take smarter, larger steps toward the solution.

### The Root-Finding Problem
Given $g(y) = 0$, find $y^*$.

**Newton-Raphson iteration:**

$$
y_{k+1} = y_k - \frac{g(y_k)}{g'(y_k)}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $y_{k+1}$ | Next iterate | Updated root approximation after Newton-Raphson step |
| $y_k$ | Current iterate | Previous root approximation |
| $g(y_k)$ | Function value at $y_k$ | Measures residual; how far current iterate is from a root |
| $g'(y_k)$ | Derivative at $y_k$ | Slope of tangent line; determines step direction and magnitude |
**Geometric meaning:** At each step, approximate $g$ by its tangent line at $y_k$, and find where the tangent crosses zero.

```python
import numpy as np

def newton_raphson(g, g_prime, y0, tol=1e-10, max_iter=100):
    """Find root of g(y) = 0 using Newton-Raphson."""
    y = y0
    for k in range(max_iter):
        gy = g(y)
        if abs(gy) < tol:
            return y
        y = y - gy / g_prime(y)
    return y

# Find sqrt(2): solve y^2 - 2 = 0
root = newton_raphson(lambda y: y**2 - 2, lambda y: 2*y, y0=1.0)
print(f"sqrt(2) ≈ {root}")  # 1.4142135623...
```

---

## 2. Newton's Method for Optimization

### From Roots to Minima
To minimize $f(\mathbf{x})$, we need to find where $\nabla f(\mathbf{x}) = \mathbf{0}$. This is a root-finding problem on the gradient. Applying Newton-Raphson to $\mathbf{g}(\mathbf{x}) = \nabla f(\mathbf{x})$:

$$
\mathbf{x}_{k+1} = \mathbf{x}_k - [\nabla^2 f(\mathbf{x}_k)]^{-1} \nabla f(\mathbf{x}_k)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{x}_{k+1}$ | Next iterate | Updated parameter vector after full Newton step |
| $\mathbf{x}_k$ | Current iterate | Parameter vector at current iteration |
| $\nabla^2 f(\mathbf{x}_k)$ | Hessian matrix at $\mathbf{x}_k$ | Matrix of second partial derivatives; encodes curvature |
| $\nabla f(\mathbf{x}_k)$ | Gradient at $\mathbf{x}_k$ | Vector of first partial derivatives; direction of steepest ascent |
where $\nabla^2 f$ is the **Hessian matrix** (Lecture 34, 39).

### Quadratic Model Interpretation
At each step, Newton's method fits a **quadratic model** to $f$ near $\mathbf{x}_k$:

$$
m_k(\mathbf{x}) = f(\mathbf{x}_k) + \nabla f(\mathbf{x}_k)^T(\mathbf{x} - \mathbf{x}_k) + \frac{1}{2}(\mathbf{x} - \mathbf{x}_k)^T \nabla^2 f(\mathbf{x}_k)(\mathbf{x} - \mathbf{x}_k)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $m_k(\mathbf{x})$ | Quadratic model at iteration $k$ | Second-order Taylor approximation of $f$ near $\mathbf{x}_k$ |
| $f(\mathbf{x}_k)$ | Function value at $\mathbf{x}_k$ | Current objective value; constant term in the model |
| $\nabla f(\mathbf{x}_k)$ | Gradient at $\mathbf{x}_k$ | First-order term; linear approximation of $f$ |
| $\nabla^2 f(\mathbf{x}_k)$ | Hessian matrix at $\mathbf{x}_k$ | Second-order term; captures curvature of $f$ |
and jumps directly to the minimum of this quadratic model.

```python
import numpy as np

def newtons_method(f, grad_f, hess_f, x0, tol=1e-10, max_iter=100):
    """Newton's method for unconstrained optimization."""
    x = x0.copy()
    
    for k in range(max_iter):
        g = grad_f(x)
        H = hess_f(x)
        
        if np.linalg.norm(g) < tol:
            break
        
        # Newton direction: solve H * d = -g
        d = np.linalg.solve(H, -g)
        x = x + d
    
    return x

# Minimize f(x) = x^4 - 8x^2 + 5
f = lambda x: np.array([x[0]**4 - 8*x[0]**2 + 5])
grad_f = lambda x: np.array([4*x[0]**3 - 16*x[0]])
hess_f = lambda x: np.array([[12*x[0]**2 - 16]])

result = newtons_method(f, grad_f, hess_f, x0=np.array([3.0]))
print(f"Minimum at x = {result[0]:.6f}")  # ≈ 2.000000
```

---

## 3. Convergence Properties

| Method | Convergence Rate | Condition |
|:---|:---|:---|
| **Gradient Descent** | Linear: $\|\mathbf{x}_{k+1} - \mathbf{x}^*\| \le c\|\mathbf{x}_k - \mathbf{x}^*\|$ | $c < 1$ |
| **Newton's Method** | **Quadratic:** $\|\mathbf{x}_{k+1} - \mathbf{x}^*\| \le C\|\mathbf{x}_k - \mathbf{x}^*\|^2$ | Near solution |

**Quadratic convergence** means the number of correct digits roughly doubles each step. Newton's method converges in ~5 iterations where gradient descent needs hundreds.

### The Catch
* Each Newton step requires computing and inverting the Hessian: $O(n^2)$ storage, $O(n^3)$ inversion.
* For $n = 10^6$ parameters (a small neural network), the Hessian has $10^{12}$ entries. **Impossible.**
* The Hessian may not be positive definite — Newton's step could go in the wrong direction.

---

## 4. Damped Newton's Method

To handle cases where the pure Newton step overshoots, add a step size:

$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{d}_k
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{x}_{k+1}$ | Next iterate | Updated parameter vector after damped Newton step |
| $\mathbf{x}_k$ | Current iterate | Parameter vector at current iteration |
| $\alpha_k$ | Step size | Controls step length along Newton direction (found via line search) |
| $\mathbf{d}_k$ | Newton direction | $-\nabla^2 f(\mathbf{x}_k)^{-1} \nabla f(\mathbf{x}_k)$; Hessian-scaled descent direction |
where $\mathbf{d}_k = -[\nabla^2 f(\mathbf{x}_k)]^{-1} \nabla f(\mathbf{x}_k)$ and $\alpha_k$ is determined by line search.

This preserves the fast convergence near the solution while being more robust far from it.

---

## 5. Quasi-Newton Methods: The Practical Compromise

Since computing the full Hessian is infeasible, **quasi-Newton methods** approximate it using only gradient information.

### BFGS (Broyden-Fletcher-Goldfarb-Shanno)
Maintains an approximation $B_k$ of the Hessian that is updated efficiently:

$$
B_{k+1} = B_k + \text{rank-2 update based on } \mathbf{s}_k, \mathbf{y}_k
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $B_{k+1}$ | Updated Hessian approximation | Next quasi-Newton estimate of the Hessian |
| $B_k$ | Current Hessian approximation | Previous quasi-Newton estimate being updated |
| $\mathbf{s}_k$ | Step in parameter space | $\mathbf{x}_{k+1} - \mathbf{x}_k$; change in parameters between iterates |
| $\mathbf{y}_k$ | Step in gradient space | $\nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k)$; change in gradients |
where $\mathbf{s}_k = \mathbf{x}_{k+1} - \mathbf{x}_k$ and $\mathbf{y}_k = \nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k)$.

* Per-iteration cost: $O(n^2)$ (vs $O(n^3)$ for full Newton)
* Convergence: Superlinear (between linear and quadratic)

### L-BFGS (Limited-memory BFGS)
Stores only the last $m$ pairs $(\mathbf{s}_k, \mathbf{y}_k)$ instead of the full $n \times n$ matrix.

* Per-iteration cost: $O(mn)$ where $m \ll n$
* The go-to method for medium-scale optimization ($n \sim 10^4$ to $10^6$)

```python
from scipy.optimize import minimize

# L-BFGS-B in action
f = lambda x: x[0]**4 - 8*x[0]**2 + 5
grad_f = lambda x: np.array([4*x[0]**3 - 16*x[0]])

result = minimize(f, x0=[3.0], jac=grad_f, method='L-BFGS-B')
print(f"Minimum at x = {result.x[0]:.6f}, f = {result.fun:.6f}")
```

---

## 6. Newton's Method in Machine Learning

### When It Works
* **Small-scale problems** ($n < 1000$): Full Newton or BFGS is practical.
* **Logistic regression:** The loss is convex with a well-behaved Hessian — Newton's method converges in ~5 iterations.
* **Natural gradient:** Uses the Fisher Information Matrix (an expected Hessian) as a preconditioner.

### When It Fails
* **Deep neural networks:** $n > 10^6$ parameters, non-convex loss, Hessian is indefinite almost everywhere.
* **Stochastic settings:** The loss is estimated from mini-batches, so the Hessian estimate is noisy.

### The Deep Learning Hierarchy

| Method | Order | Hessian | Scalability | Use Case |
|:---|:---|:---|:---|:---|
| **SGD** | 1st | None | $O(n)$ | Deep learning |
| **Adam** | 1st | None | $O(n)$ | Deep learning (default) |
| **L-BFGS** | Quasi-2nd | Approximate | $O(mn)$ | Medium-scale convex |
| **Newton** | 2nd | Full | $O(n^3)$ | Small-scale convex |

```python
# Comparison: GD vs Newton on a quadratic
import numpy as np

# f(x) = 0.5 * x^T A x - b^T x
A = np.array([[4, 2], [2, 6]])
b = np.array([1, 1])
f = lambda x: 0.5 * x @ A @ x - b @ x
grad_f = lambda x: A @ x - b
hess_f = lambda: A

# Gradient Descent
x_gd = np.array([5.0, 5.0])
lr = 0.1
for _ in range(100):
    x_gd -= lr * grad_f(x_gd)

# Newton's Method
x_newt = np.array([5.0, 5.0])
for _ in range(10):
    x_newt -= np.linalg.solve(hess_f(), grad_f(x_newt))

print(f"GD (100 iters):   {x_gd}, f = {f(x_gd):.8f}")
print(f"Newton (10 iters): {x_newt}, f = {f(x_newt):.8f}")
# Newton converges to exact solution in 1 step for quadratics
```

---

## 7. Summary

| Concept | Key Takeaway |
|:---|:---|
| **Newton's method** | Uses Hessian for quadratic convergence |
| **Quadratic model** | Fits a parabola, jumps to its minimum |
| **Strength** | Very fast convergence near solution |
| **Weakness** | $O(n^3)$ Hessian inversion, may diverge far from solution |
| **BFGS/L-BFGS** | Approximate Hessian, practical compromise |
| **Deep learning** | Too expensive — use first-order methods (SGD, Adam) |

> **Check your intuition:** If you apply Newton's method to a quadratic function, how many iterations does it take to converge exactly? *(Answer: Exactly 1. A quadratic function's Taylor expansion is exact, so the quadratic model IS the function — Newton jumps directly to the minimum.)*

---

## 8. Penalty Function Method

### Motivation and Intuition
Newton's method solves unconstrained problems. But what if you have constraints? The **Penalty Function Method** converts a constrained optimization problem into a sequence of unconstrained problems by adding a **penalty term** that penalizes constraint violations. As the penalty parameter grows, the solution of the unconstrained problem approaches the solution of the constrained problem.

### The Idea
Transform:

$$
\min f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) \le 0
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $f(\mathbf{x})$ | Objective function | Quantity to be minimized |
| $\mathbf{x}$ | Decision variable vector | Parameters to optimize over |
| $g_i(\mathbf{x})$ | $i$-th inequality constraint | Must be $\le 0$ for feasibility; defines the feasible region |
into a sequence of unconstrained problems:

$$
\min \; P(\mathbf{x}, \mu) = f(\mathbf{x}) + \mu \sum_{i=1}^{m} \left[\max(0, g_i(\mathbf{x}))\right]^2
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $P(\mathbf{x}, \mu)$ | Penalized objective | Unconstrained surrogate combining original objective and penalty |
| $f(\mathbf{x})$ | Original objective | Function to be minimized |
| $\mu$ | Penalty parameter | Weight on constraint violations; increased over iterations |
| $\max(0, g_i(\mathbf{x}))$ | $i$-th constraint violation | Zero when feasible, positive when $g_i$ is violated |
| $[\cdot]^2$ | Squared penalty operator | Ensures differentiability at $g_i = 0$ |
where $\mu > 0$ is the **penalty parameter**. As $\mu \to \infty$, any constraint violation is heavily penalized, so the minimizer of $P$ approaches the minimizer of the original constrained problem.

### Why Squared?
The penalty term $\left[\max(0, g_i(\mathbf{x}))\right]^2$ is differentiable everywhere (including at $g_i = 0$), which allows us to apply gradient-based methods like Newton's method on the penalized objective.

### Algorithm

1. Start with an initial penalty parameter $\mu_0 > 0$ (e.g., $\mu_0 = 1$).
2. Solve the unconstrained problem: $\mathbf{x}_k = \arg\min_\mathbf{x} P(\mathbf{x}, \mu_k)$.
3. Increase the penalty: $\mu_{k+1} = \beta \mu_k$ (e.g., $\beta = 10$).
4. Use $\mathbf{x}_k$ as the starting point for the next solve.
5. Repeat until constraint violation is sufficiently small.

```python
import numpy as np
from scipy.optimize import minimize

def penalty_method(f, g_funcs, x0, mu_init=1.0, beta=10.0, tol=1e-6, max_iter=50):
    """
    Penalty method for constrained optimization.
    min f(x) s.t. g_i(x) <= 0 for all i.
    """
    mu = mu_init
    x = x0.copy()
    
    for k in range(max_iter):
        # Define penalized objective
        def penalized(x, mu=mu):
            penalty = sum(max(0, g(x))**2 for g in g_funcs)
            return f(x) + mu * penalty
        
        # Solve unconstrained subproblem
        result = minimize(penalized, x, method='L-BFGS-B')
        x = result.x
        
        # Check constraint violation
        violation = max(abs(g(x)) for g in g_funcs if g(x) > 0) if any(g(x) > 0 for g in g_funcs) else 0
        
        print(f"Iter {k}: mu={mu:.1f}, violation={violation:.2e}, f={f(x):.6f}")
        
        if violation < tol:
            break
        
        mu *= beta
    
    return x
```

### Worked Example

**Problem:**

$$
\min x^2 + y^2 \quad \text{s.t.} \quad x + y \ge 1
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $x^2 + y^2$ | Objective function | Euclidean distance squared from origin (to be minimized) |
| $x + y \ge 1$ | Inequality constraint | Requires solution to lie above or on the line $x + y = 1$ |
| $x, y$ | Decision variables | Two-dimensional parameters to optimize over |
Rewrite constraint: $g(x,y) = 1 - x - y \le 0$.

```python
import numpy as np
from scipy.optimize import minimize

# Original problem
f = lambda v: v[0]**2 + v[1]**2
g = lambda v: 1 - v[0] - v[1]  # g <= 0 means x + y >= 1

# Penalty method
mu = 1.0
x = np.array([0.0, 0.0])

for iteration in range(20):
    def penalized(v, mu=mu):
        return f(v) + mu * max(0, g(v))**2
    
    result = minimize(penalized, x, method='L-BFGS-B')
    x = result.x
    violation = max(0, g(x))
    
    print(f"Iter {iteration}: x={x}, violation={violation:.4e}")
    
    if violation < 1e-8:
        break
    mu *= 10

print(f"\nOptimal: x={x}, f={f(x):.6f}")
# Should converge to [0.5, 0.5], f = 0.5
```

### Quadratic Penalty vs Exact Penalty

| Method | Penalty Function | Properties |
|:---|:---|:---|
| **Quadratic penalty** | $\mu \sum [\max(0, g_i)]^2$ | Smooth, but $\mu \to \infty$ causes ill-conditioning |
| **$\ell_1$ exact penalty** | $\mu \sum \max(0, g_i)$ | Non-smooth, but finite $\mu$ gives exact solution |
| **Augmented Lagrangian** | Combines penalty + Lagrange multipliers | Best of both worlds — no ill-conditioning |

### Limitation
As $\mu \to \infty$, the penalized objective becomes extremely **ill-conditioned** — the Hessian develops very large and very small eigenvalues, making Newton's method slow. This is why the **Augmented Lagrangian Method** (which adds Lagrange multipliers to the penalty) is preferred in practice.

---

## 9. Summary: Newton + Penalty Combined

For a constrained problem, the typical approach is:

1. **Penalty method:** Convert constrained → sequence of unconstrained problems.
2. **Newton/L-BFGS:** Solve each unconstrained subproblem efficiently.
3. **Increase penalty:** Move to the next subproblem with tighter constraints.

This combination is the foundation of **interior-point methods**, which are the dominant algorithm for convex constrained optimization (including SVMs and logistic regression with constraints).

> **Check your intuition:** Why does the penalty method need $\mu \to \infty$ for an exact solution, and why is this problematic? *(Answer: For any finite $\mu$, the penalized solution still has a small constraint violation. But as $\mu$ grows, the condition number of the Hessian grows proportionally, making Newton's steps numerically unstable. The augmented Lagrangian method fixes this by adding explicit Lagrange multipliers.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 44: Steepest Descent Method](Lecture%2044%20Steepest%20Descent%20Method.md) — provides the first-order methods that Newton's method improves upon with second-order curvature information
- **Next:** [Lecture 46: Python Implementation of Convex Optimization](Lecture%2046%20Python%20Implementation%20of%20Convex%20Optimization.md) — implements these optimization methods in Python for practical convex problems
- **Related:** [Lecture 41: Unconstrained Optimization](Lecture%2041%20Unconstrained%20Optimization.md) — Newton's method solves unconstrained problems; penalty methods convert constrained to unconstrained
- **Related:** [Lecture 42: Constrained Optimization-I](Lecture%2042%20Constrained%20Optimization-I.md) — KKT conditions provide the theoretical foundation that penalty methods approximate
- **Related:** [Lecture 44: Steepest Descent Method](Lecture%2044%20Steepest%20Descent%20Method.md) — gradient descent is the practical alternative when Newton's method is too expensive
