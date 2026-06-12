## Limits and Continuity

*Essential Mathematics for ML — Structured Notes*

---

## 1. Limit of a Function

### Motivation and Intuition
When we train a neural network, we care about what happens as the learning rate approaches zero, as the number of layers approaches infinity, or as the batch size grows without bound. The mathematical tool that lets us reason about "what happens as something approaches a value" is the **limit**. Without limits, we cannot define derivatives, and without derivatives, we cannot do gradient descent.

### Formal Definition
Let $f: X \to Y$ where $X, Y \subseteq \mathbb{R}$. We write:

$$
\lim_{x \to a} f(x) = L
$$

if and only if the **left-hand limit** and **right-hand limit** both exist and are equal:

$$
\lim_{x \to a^-} f(x) = \lim_{x \to a^+} f(x) = L
$$

**Key distinction:** The limit describes what $f(x)$ approaches as $x$ gets arbitrarily close to $a$ — it says nothing about $f(a)$ itself. The function may not even be defined at $x = a$.

---

## 2. Geometric Interpretation

Graphically, $\lim_{x \to a} f(x) = L$ means that as you trace the curve of $f$ from both the left side and the right side toward $x = a$, the $y$-values converge to the same height $L$.

* If the left and right limits differ, the limit **does not exist** at that point.
* If the curve has a "hole" at $x = a$ but the surrounding values approach $L$, the limit still exists — the hole is irrelevant.

---

## 3. Algebraic Properties of Limits

Given $\lim_{x \to a} f(x) = L$ and $\lim_{x \to a} g(x) = M$:

| Property | Rule |
|:---|:---|
| **Sum** | $\lim_{x \to a} [f(x) + g(x)] = L + M$ |
| **Difference** | $\lim_{x \to a} [f(x) - g(x)] = L - M$ |
| **Product** | $\lim_{x \to a} [f(x) \cdot g(x)] = L \cdot M$ |
| **Quotient** | $\lim_{x \to a} \frac{f(x)}{g(x)} = \frac{L}{M}, \quad M \neq 0$ |
| **Scalar Multiple** | $\lim_{x \to a} [c \cdot f(x)] = c \cdot L$ |

```python
import numpy as np

# Numerical limit approximation
def numerical_limit(f, a, h=1e-10):
    """Approximate the limit of f(x) as x -> a."""
    left = f(a - h)
    right = f(a + h)
    if np.isclose(left, right):
        return (left + right) / 2
    return None  # limit does not exist

f = lambda x: x**2
print(numerical_limit(f, 3))  # ~9.0
```

---

## 4. Limits That Do Not Exist

### Jump Discontinuity
The left and right limits exist but differ:

$$
\lim_{x \to a^-} f(x) \neq \lim_{x \to a^+} f(x)
$$

**Example:** The step function $\text{sgn}(x)$ jumps from $-1$ to $+1$ at $x = 0$.

### Oscillatory Behavior
The function oscillates infinitely often near $a$, never settling on a single value:

$$
f(x) = \sin\left(\frac{1}{x}\right) \quad \text{as } x \to 0
$$

### Unbounded Growth
The function grows without bound:

$$
\lim_{x \to 0} \frac{1}{x^2} = \infty
$$

**Deep Learning Failure Mode:** During training, if a weight grows without bound, the pre-activation $z = W\mathbf{x} + \mathbf{b}$ can produce $\sin(1/z)$-style oscillations in gradient signals, causing the optimizer to oscillate wildly and never converge. Gradient clipping exists precisely to prevent this.

---

## 5. Continuity

### Definition
A function $f$ is **continuous** at $x = a$ if all three conditions hold:

1. $f(a)$ is defined
2. $\lim_{x \to a} f(x)$ exists
3. $\lim_{x \to a} f(x) = f(a)$

If $f$ is continuous at every point in its domain, we say $f$ is a **continuous function**.

### Intuition
A function is continuous if you can draw its graph without lifting your pen. There are no holes, jumps, or vertical asymptotes.

---

## 6. Types of Discontinuity

### Removable Discontinuity
The limit exists, but $f(a)$ is either undefined or defined to be a different value. "Patching" the function at that single point makes it continuous.

$$
f(x) = \frac{x^2 - 1}{x - 1} \quad \text{has a removable discontinuity at } x = 1
$$

The limit is $2$, but $f(1)$ is undefined.

### Jump Discontinuity
The left and right limits exist but are different. No amount of patching a single point can fix this.

### Infinite Discontinuity
The function has a vertical asymptote — the limit is $\pm\infty$.

```python
# Demonstrating removable discontinuity
import numpy as np

def f(x):
    if np.isclose(x, 1):
        return None  # undefined at x=1
    return (x**2 - 1) / (x - 1)

# But the limit exists:
# lim_{x->1} (x^2-1)/(x-1) = lim_{x->1} (x+1) = 2
```

---

## 7. Continuity and Machine Learning

### Why Continuity Matters

Loss functions must be continuous (and typically differentiable) for gradient descent to work. If the loss surface has discontinuities, the gradient is undefined at those points, and the optimizer has no information about which direction to move.

**Deep Learning Failure Mode:** The Heaviside step function $H(x)$ is discontinuous at $x = 0$. If used as an activation function, its gradient is zero everywhere except at $x = 0$, where it is undefined. This is why sigmoid and ReLU replaced it — they are continuous (and ReLU is differentiable almost everywhere).

### The Sigmoid Function: A Case Study in Continuity

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

* **Continuous?** Yes — smooth curve with no breaks.
* **Differentiable?** Yes — everywhere, with derivative $\sigma'(x) = \sigma(x)(1 - \sigma(x))$.
* **Bounded?** Yes — output always in $(0, 1)$.

This is precisely why it became the standard activation function for decades.

---

## 8. The Intermediate Value Theorem

If $f$ is continuous on $[a, b]$ and $k$ is any value between $f(a)$ and $f(b)$, then there exists at least one $c \in (a, b)$ such that $f(c) = k$.

**ML Connection:** This theorem guarantees that if a neural network's output layer produces values in $[0, 1]$ (via sigmoid) and the target is $0.7$, there exists some input that would produce exactly $0.7$. The network's output space is "connected" — there are no unreachable values between the extremes.

---

## 9. Summary Table

| Concept | Definition | ML Relevance |
|:---|:---|:---|
| **Limit** | Value $f(x)$ approaches as $x \to a$ | Foundation of derivatives |
| **Continuous** | Limit equals function value | Required for gradient descent |
| **Removable discontinuity** | Limit exists but $f(a) \neq$ limit | Patchable with normalization |
| **Jump discontinuity** | Left $\neq$ right limit | Step functions (avoided in DL) |
| **Infinite discontinuity** | Vertical asymptote | Exploding gradients |

> **Check your intuition:** Is ReLU continuous at $x = 0$? *(Answer: Yes. $\lim_{x \to 0^-} \max(0,x) = 0$ and $\lim_{x \to 0^+} \max(0,x) = 0$, and $\text{ReLU}(0) = 0$. All three conditions are satisfied. It is continuous but not differentiable at $x = 0$.)*
