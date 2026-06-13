## Constrained Optimization and KKT Conditions

*Essential Mathematics for ML — Structured Notes*

---

## 1. The Constrained Optimization Problem

### Motivation and Intuition
In many ML problems, we cannot freely choose any parameter values. SVMs require the margin to be at least 1. Regularized models constrain the weight norm. Resource allocation problems have budget constraints. **Constrained optimization** adds restrictions to the feasible region, and the **KKT conditions** are the mathematical framework that characterizes optimal solutions under these constraints.

### Formal Definition
$$
\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x}) \quad \text{subject to} \quad g_i(\mathbf{x}) \le 0, \; i = 1, \dots, m
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\mathbf{x} \in \mathbb{R}^n$ | Decision variables (model parameters) | The vector we optimize over; represents all learnable parameters |
| $f(\mathbf{x})$ | Objective function | The loss or cost function we want to minimize (e.g., empirical risk) |
| $g_i(\mathbf{x}) \le 0$ | Inequality constraints | Restrictions on feasible solutions (e.g., margin $\ge$ 1, norm $\le$ budget) |
| $i = 1, \dots, m$ | Constraint index | $m$ constraints define the feasible region $C = \{\mathbf{x} : g_i(\mathbf{x}) \le 0\}$ |

* $f(\mathbf{x})$: **Objective function** (the loss we want to minimize)
* $g_i(\mathbf{x}) \le 0$: **Inequality constraints** (restrictions on the parameters)
* The set $C = \{\mathbf{x} : g_i(\mathbf{x}) \le 0\}$ is the **feasible region**

### Classification of Constrained Problems

| Type | Objective | Constraints |
|:---|:---|:---|
| **Linear Programming (LP)** | Linear | Linear inequalities |
| **Quadratic Programming (QP)** | Quadratic | Linear inequalities |
| **Convex Programming** | Convex | Convex inequalities |
| **General Nonlinear** | Nonlinear | Nonlinear inequalities |

**ML Examples:**
* **SVM:** QP — minimize $\frac{1}{2}\|\mathbf{w}\|^2$ subject to $y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1$
* **Lasso:** Convex — minimize $\|X\mathbf{w} - \mathbf{y}\|^2$ subject to $\|\mathbf{w}\|_1 \le t$
* **Ridge:** Convex — minimize $\|X\mathbf{w} - \mathbf{y}\|^2$ subject to $\|\mathbf{w}\|_2^2 \le t$

---

## 2. Lagrange Multipliers (Equality Constraints)

### Problem
$$
\min f(\mathbf{x}) \quad \text{subject to} \quad h_j(\mathbf{x}) = 0, \; j = 1, \dots, p
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $f(\mathbf{x})$ | Objective function | The function to minimize (e.g., loss, energy) |
| $h_j(\mathbf{x}) = 0$ | Equality constraints | Hard constraints that must be exactly satisfied (e.g., $\sum w_i = 1$) |
| $j = 1, \dots, p$ | Constraint index | $p$ equality constraints define the feasible manifold |

### The Method
Define the **Lagrangian**:

$$
\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \sum_{j=1}^{p} \lambda_j h_j(\mathbf{x})
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda})$ | Lagrangian function | Combines objective and constraints into a single unconstrained function |
| $f(\mathbf{x})$ | Original objective | The function we want to minimize |
| $\lambda_j$ | Lagrange multipliers | Sensitivity of optimal value to constraint $j$; "shadow prices" |
| $h_j(\mathbf{x})$ | Equality constraint functions | Must equal zero at the solution |

**Necessary condition:** At the optimal point $(\bar{\mathbf{x}}, \bar{\boldsymbol{\lambda}})$:

$$
\nabla_\mathbf{x} \mathcal{L} = \mathbf{0}, \quad \nabla_{\boldsymbol{\lambda}} \mathcal{L} = \mathbf{0}
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $\nabla_\mathbf{x} \mathcal{L} = \mathbf{0}$ | Stationarity w.r.t. $\mathbf{x}$ | Gradient of Lagrangian vanishes; optimal $\mathbf{x}$ balances objective and constraints |
| $\nabla_{\boldsymbol{\lambda}} \mathcal{L} = \mathbf{0}$ | Constraint satisfaction | Recovers original constraints $h_j(\mathbf{x}) = 0$ |
| $n + p$ equations | System size | $n$ stationarity + $p$ constraint equations for $n$ variables + $p$ multipliers |

This gives $n + p$ equations in $n + p$ unknowns.

**Intuition:** The Lagrange multipliers $\lambda_j$ measure the "price" of each constraint — how much the objective would improve if the constraint were relaxed by one unit.

```python
import sympy as sp

# Minimize f(x,y) = x^2 + y^2 subject to x + y = 1
x, y, lam = sp.symbols('x y lambda')

f = x**2 + y**2
h = x + y - 1

# Lagrangian
L = f + lam * h

# Solve grad L = 0
eq1 = sp.diff(L, x)  # 2x + lambda = 0
eq2 = sp.diff(L, y)  # 2y + lambda = 0
eq3 = sp.diff(L, lam) # x + y - 1 = 0

sol = sp.solve([eq1, eq2, eq3], (x, y, lam))
print(f"Optimal: x={sol[x]}, y={sol[y]}, lambda={sol[lambda]}")
# x=1/2, y=1/2, lambda=-1
```

---

## 3. KKT Conditions (Inequality Constraints)

### The Full Problem
$$
\min f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) \le 0, \; i = 1, \dots, m
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\min f(\mathbf{x})$ | Minimize objective $f$ | Find the parameters $\mathbf{x}$ that make $f$ as small as possible |
| $\mathbf{x}$ | Decision variables | The parameters we optimize (e.g., SVM weights) |
| $g_i(\mathbf{x}) \le 0$ | $i$-th inequality constraint | Defines a restriction: feasible points must satisfy all $m$ inequalities |
| $i = 1, \dots, m$ | Constraint index | There are $m$ inequality constraints defining the feasible region |
| Feasible region $C$ | $\{\mathbf{x} : g_i(\mathbf{x}) \le 0, \forall i\}$ | The set of all points satisfying every constraint |

### The KKT Conditions
At a local minimum $\bar{\mathbf{x}}$ (with constraint qualification), there exist **KKT multipliers** $\bar{\lambda}_i \ge 0$ such that:

| Condition | Formula | Name |
|:---|:---|:---|
| 1 | $\nabla f(\bar{\mathbf{x}}) + \sum_{i=1}^m \bar{\lambda}_i \nabla g_i(\bar{\mathbf{x}}) = \mathbf{0}$ | Stationarity |
| 2 | $g_i(\bar{\mathbf{x}}) \le 0, \; \forall i$ | Primal feasibility |
| 3 | $\bar{\lambda}_i \ge 0, \; \forall i$ | Dual feasibility |
| 4 | $\bar{\lambda}_i g_i(\bar{\mathbf{x}}) = 0, \; \forall i$ | Complementary slackness |

### Complementary Slackness Explained
This is the most powerful condition: for each constraint $i$, either:

* $\bar{\lambda}_i = 0$ (the constraint is **inactive** — it doesn't affect the solution), OR
* $g_i(\bar{\mathbf{x}}) = 0$ (the constraint is **active** — it binds at the solution)

**ML Connection:** In SVMs, complementary slackness tells us that only the support vectors (data points on the margin) have non-zero Lagrange multipliers. All other data points have $\lambda_i = 0$ and don't affect the decision boundary.

---

## 4. Constraint Qualification

The KKT conditions require a **constraint qualification (CQ)** — an additional technical condition ensuring the KKT multipliers exist with $\lambda_0 \neq 0$ (the multiplier on the objective function).

### Common CQs

* **Linear Independence CQ (LICQ):** The gradients of the active constraints are linearly independent.
* **Slater's Condition (for convex problems):** There exists a strictly feasible point $\mathbf{x}_0$ where $g_i(\mathbf{x}_0) < 0$ for all $i$.

**Deep Learning Connection:** In practice, most ML problems satisfy Slater's condition (you can always find a point strictly inside the feasible region), so the KKT conditions are directly applicable.

---

## 5. KKT as Sufficient Condition for Convex Problems

### Theorem
If $f$ and $g_i$ are all **convex** and differentiable, then any point satisfying the KKT conditions is a **global minimum**.

This is the constrained analog of "local = global" for convex functions.

**Why this is huge:** For convex optimization problems (SVM, Lasso, Ridge), we don't need to search for the solution — we just solve the KKT equations.

```python
import numpy as np
from scipy.optimize import minimize

# Solve: minimize x^2 + y^2 subject to x + y >= 1
def objective(x):
    return x[0]**2 + x[1]**2

def constraint(x):
    return x[0] + x[1] - 1  # g(x) >= 0, i.e., -g(x) <= 0

cons = {'type': 'ineq', 'fun': constraint}
result = minimize(objective, x0=[0, 0], constraints=cons)
print(f"Optimal: x={result.x}, f={result.fun}")
# x ≈ [0.5, 0.5], f ≈ 0.5
```

---

## 6. Worked Example: SVM-Style Problem

**Problem:**

$$
\min \frac{1}{2}(x^2 + y^2) \quad \text{s.t.} \quad x + 2y \ge 3
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $f(x,y) = \frac{1}{2}(x^2 + y^2)$ | Objective function | Quadratic function to minimize — analogous to minimizing $\frac{1}{2}\|\mathbf{w}\|^2$ in SVM |
| $\min$ | Minimization | We search for the smallest possible value of $f$ |
| $x + 2y \ge 3$ | Inequality constraint | The solution must lie in the half-plane where $x + 2y \ge 3$ |
| $\text{s.t.}$ | Subject to | Indicates the constraint that restricts the feasible region |

Rewrite constraint: $g(x,y) = 3 - x - 2y \le 0$.

**Lagrangian:** $\mathcal{L} = \frac{1}{2}(x^2 + y^2) + \lambda(3 - x - 2y)$

**KKT Conditions:**

1. **Stationarity:**
   * $\frac{\partial \mathcal{L}}{\partial x} = x - \lambda = 0 \implies x = \lambda$
   * $\frac{\partial \mathcal{L}}{\partial y} = y - 2\lambda = 0 \implies y = 2\lambda$

2. **Complementary slackness:** $\lambda(3 - x - 2y) = 0$

3. **Dual feasibility:** $\lambda \ge 0$

4. **Primal feasibility:** $x + 2y \ge 3$

**Case 1: $\lambda = 0$** → $x = 0, y = 0$. Check feasibility: $0 + 0 \ge 3$? No. Infeasible.

**Case 2: $3 - x - 2y = 0$** → $3 - \lambda - 4\lambda = 0 \implies \lambda = 3/5$.

Solution: $x = 3/5, y = 6/5, \lambda = 3/5$.

Check: $x + 2y = 3/5 + 12/5 = 15/5 = 3$. ✓

```python
import numpy as np

# Verify
x, y, lam = 3/5, 6/5, 3/5
print(f"x={x}, y={y}, lambda={lam}")
print(f"Constraint: {x + 2*y} >= 3? {x + 2*y >= 3}")
print(f"Stationarity: x-lam={x-lam}, y-2*lam={y-2*lam}")
print(f"Objective: {0.5*(x**2 + y**2)}")
```

---

## 7. Summary

| Condition | Meaning |
|:---|:---|
| **Stationarity** | Gradient of Lagrangian = 0 |
| **Primal feasibility** | Constraints are satisfied |
| **Dual feasibility** | Multipliers are non-negative |
| **Complementary slackness** | Either $\lambda_i = 0$ or constraint is active |
| **Convex + KKT** | KKT conditions are sufficient for global optimality |

> **Check your intuition:** In an SVM, why do most Lagrange multipliers $\lambda_i$ equal zero? *(Answer: By complementary slackness, $\lambda_i = 0$ for all points NOT on the margin. Only the support vectors (points exactly on the margin) have $\lambda_i > 0$. This is why SVMs are memory-efficient — only support vectors matter.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 41: Unconstrained Optimization](Lecture%2041%20Unconstrained%20Optimization.md) — provides the unconstrained foundation that constrained optimization extends with KKT conditions
- **Next:** [Lecture 43: Constrained Optimization-II](Lecture%2043%20Constrained%20Optimization-II.md) — covers numerical algorithms for solving constrained problems
- **Related:** [Lecture 41: Unconstrained Optimization](Lecture%2041%20Unconstrained%20Optimization.md) — the unconstrained framework that penalty methods and KKT build upon
- **Related:** [Lecture 45: Newton's and Penalty Function Method](Lecture%2045%20Newton%27s%20and%20Penalty%20Function%20Method.md) — penalty methods convert constrained problems to unconstrained subproblems
- **Related:** [Lecture 56: Duality and the Dual Problem](Lecture%2056%20Duality%20and%20the%20Dual%20Problem.md) — KKT conditions connect to the dual formulation of constrained problems
