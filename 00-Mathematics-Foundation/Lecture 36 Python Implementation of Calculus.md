## Python Implementation of Calculus

*Essential Mathematics for ML — Structured Notes*

---

## 1. Why SymPy?

### Motivation and Intuition
We have spent several lectures deriving derivatives, computing gradients, and working through chain rules by hand. In practice, no ML engineer manually differentiates a 50-layer neural network. We rely on autodiff frameworks (PyTorch, TensorFlow) that automate the chain rule. But for **understanding, debugging, and prototyping**, being able to symbolically verify calculus computations is invaluable. **SymPy** is Python's symbolic mathematics library — it manipulates expressions algebraically, not numerically, giving us exact results.

### Setup

```python
import sympy as sp

# Declare symbolic variables
x, y, z = sp.symbols('x y z')

# SymPy expressions are exact — no floating point
expr = x**2 + sp.sin(x)
print(expr)  # x**2 + sin(x)
```

---

## 2. Differentiation with SymPy

### Single Variable

```python
import sympy as sp

x = sp.Symbol('x')

# First derivative of sin(x)
f = sp.sin(x)
f_prime = sp.diff(f, x)
print(f_prime)  # cos(x)

# First derivative of sin(x^2)
g = sp.sin(x**2)
g_prime = sp.diff(g, x)
print(g_prime)  # 2*x*cos(x**2)

# Second derivative
g_double_prime = sp.diff(g, x, 2)
print(g_double_prime)  # -4*x**2*sin(x**2) + 2*cos(x**2)
```

### Multivariable (Partial Derivatives)

```python
import sympy as sp

x, y = sp.symbols('x y')

# Function of two variables
f = x**3 * y + sp.sin(x * y)

# Partial derivative with respect to x
df_dx = sp.diff(f, x)
print(df_dx)  # 3*x**2*y + y*cos(x*y)

# Partial derivative with respect to y
df_dy = sp.diff(f, y)
print(df_dy)  # x**3 + x*cos(x*y)

# Mixed partial (Clairaut's theorem: should be equal)
d2f_dxdy = sp.diff(f, x, y)
d2f_dydx = sp.diff(f, y, x)
print(d2f_dxdy == d2f_dydx)  # True
```

### Gradient Vector

```python
import sympy as sp

x, y, z = sp.symbols('x y z')
f = x**2 + y**2 + z**2

# Gradient as a matrix
gradient = sp.Matrix([sp.diff(f, var) for var in (x, y, z)])
print(gradient)  # Matrix([[2*x], [2*y], [2*z]])
```

---

## 3. Integration with SymPy

### Indefinite Integrals

```python
import sympy as sp

x = sp.Symbol('x')

# Indefinite integral of x^2
f = x**2
F = sp.integrate(f, x)
print(F)  # x**3/3

# Indefinite integral of e^x * sin(x)
g = sp.exp(x) * sp.sin(x)
G = sp.integrate(g, x)
print(G)  # exp(x)*sin(x)/2 - exp(x)*cos(x)/2
```

### Definite Integrals

```python
import sympy as sp

x = sp.Symbol('x')

# Definite integral of x^2 from 0 to 1
result = sp.integrate(x**2, (x, 0, 1))
print(result)  # 1/3

# Double integral of x*y over [0,1] x [0,1]
x, y = sp.symbols('x y')
result = sp.integrate(x*y, (x, 0, 1), (y, 0, 1))
print(result)  # 1/4
```

### Numerical Integration

For integrals that have no closed-form solution, use `sp.N()` or `scipy`:

```python
import sympy as sp
from scipy import integrate
import numpy as np

# Numerical integration via scipy
f_num = lambda x: np.sin(x**2)
result, error = integrate.quad(f_num, 0, np.pi)
print(f"Result: {result:.6f}, Error: {error:.2e}")
```

---

## 4. Limits with SymPy

```python
import sympy as sp

x = sp.Symbol('x')

# lim_{x->0} sin(x)/x = 1
limit1 = sp.limit(sp.sin(x)/x, x, 0)
print(limit1)  # 1

# lim_{x->inf} (1 + 1/x)^x = e
limit2 = sp.limit((1 + 1/x)**x, x, sp.oo)
print(limit2)  # exp(1)

# lim_{x->0+} 1/x = infinity
limit3 = sp.limit(1/x, x, 0, '+')
print(limit3)  # oo
```

---

## 5. Series Expansion (Taylor Series)

Taylor series approximate functions as polynomials near a point. They are the foundation of numerical optimization — Newton's method uses the second-order Taylor expansion.

```python
import sympy as sp

x = sp.Symbol('x')

# Taylor expansion of sin(x) around x=0, order 5
taylor_sin = sp.series(sp.sin(x), x, 0, n=6)
print(taylor_sin)  # x - x**3/6 + x**5/120 + O(x**6)

# Taylor expansion of e^x around x=0, order 4
taylor_exp = sp.series(sp.exp(x), x, 0, n=5)
print(taylor_exp)  # 1 + x + x**2/2 + x**3/6 + x**4/24 + O(x**5)

# Remove the O() term for algebraic manipulation
taylor_sin_clean = taylor_sin.removeO()
print(taylor_sin_clean)  # x**5/120 - x**3/6 + x
```

### ML Connection: Newton's Method

The second-order Taylor expansion of a loss function $\mathcal{L}(\theta)$ around $\theta_t$:

$$
\mathcal{L}(\theta) \approx \mathcal{L}(\theta_t) + \nabla \mathcal{L}^T(\theta - \theta_t) + \frac{1}{2}(\theta - \theta_t)^T H (\theta - \theta_t)
$$

Minimizing this quadratic approximation gives Newton's update rule:

$$
\theta_{t+1} = \theta_t - H^{-1} \nabla \mathcal{L}
$$

where $H$ is the Hessian matrix. This converges faster than gradient descent but requires computing and inverting $H$ — which is $O(n^3)$ and impractical for large models.

```python
import sympy as sp

x = sp.Symbol('x')

# f(x) = x^4 - 3x^2 + 2x
f = x**4 - 3*x**2 + 2*x

# Taylor expansion around x=1, order 3
taylor_f = sp.series(f, x, 1, n=4).removeO()
print(f"f(x) ≈ {taylor_f}")

# The coefficient of (x-1)^2 gives half the second derivative (Hessian in 1D)
# f''(1)/2 = coefficient of (x-1)^2
```

---

## 6. Solving Equations

```python
import sympy as sp

x = sp.Symbol('x')

# Solve x^2 - 4 = 0
solutions = sp.solve(x**2 - 4, x)
print(solutions)  # [-2, 2]

# Solve system of equations
y = sp.Symbol('y')
eq1 = sp.Eq(x + y, 5)
eq2 = sp.Eq(x - y, 1)
sol = sp.solve([eq1, eq2], (x, y))
print(sol)  # {x: 3, y: 2}

# Find critical points: set derivative to zero
f = x**3 - 3*x**2 + 2
f_prime = sp.diff(f, x)
critical = sp.solve(f_prime, x)
print(critical)  # [0, 2]
```

---

## 7. SymPy vs Autodiff: When to Use What

| Tool | Approach | Use Case |
|:---|:---|:---|
| **SymPy** | Symbolic (exact) | Proving theorems, verifying derivatives, prototyping |
| **PyTorch/TensorFlow** | Autodiff (numerical) | Training large models, backpropagation at scale |
| **NumPy/SciPy** | Numerical (finite diff) | Quick sanity checks, debugging gradients |

**Deep Learning Connection:** Every deep learning framework uses **automatic differentiation** — it applies the chain rule numerically through the computation graph, not symbolically. SymPy helps you understand *what* the chain rule is computing; PyTorch executes it at scale.

```python
# PyTorch autograd: chain rule in action
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x**4 - 3*x**2 + 2*x
y.backward()
print(f"dy/dx at x=3: {x.grad}")  # 4*27 - 6*3 = 90

# Verify with SymPy
import sympy as sp
xs = sp.Symbol('x')
f_prime = sp.diff(xs**4 - 3*xs**2 + 2*xs, xs)
print(f"SymPy: {f_prime.subs(xs, 3)}")  # 90
```

> **Check your intuition:** Why can't SymPy differentiate a PyTorch neural network? *(Answer: SymPy works on symbolic expressions — it needs a closed-form formula. A neural network with stochastic activation functions and learned weights is a numerical computation graph, not a simple algebraic expression. Autodiff is the correct tool for that.)*
