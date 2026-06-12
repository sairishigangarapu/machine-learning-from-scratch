## Functions

*Essential Mathematics for ML — Structured Notes*

---

## 1. Definition of a Function

### Motivation and Intuition
Every machine learning model is fundamentally a function. A neural network takes in a vector of pixel values (input) and produces a probability distribution over classes (output). A linear regression model takes in features and outputs a continuous prediction. Before we can understand gradient descent, loss functions, or backpropagation, we need a rigorous understanding of what a function actually is.

### Formal Definition
Let $A$ and $B$ be two non-empty sets. A **function** $f: A \to B$ is a rule or correspondence that associates each element of $A$ to exactly one element of $B$.

Two non-negotiable conditions:

1. **Total:** Every element in $A$ must have an image in $B$. No element is left behind.
2. **Unique:** Each element in $A$ maps to one and only one element in $B$. No element splits into two outputs.

```python
# A function in Python is literally a function
def f(x):
    return x**2 + 1

# Every input produces exactly one output
print(f(3))  # 10 — unique output
```

---

## 2. Domain, Codomain, and Range

### Definitions

* **Domain ($A$):** The set of all valid inputs. In ML, this is typically $\mathbb{R}^n$ (feature space).
* **Codomain ($B$):** The set of all possible outputs the function is defined to produce.
* **Range ($f(A)$):** The set of all outputs actually produced by $f$. The range is always a subset of the codomain: $f(A) \subseteq B$.

### Example

Let $f: \mathbb{R} \to \mathbb{R}$ defined by $f(x) = x^2$.

* **Domain:** $\mathbb{R}$ (all real numbers)
* **Codomain:** $\mathbb{R}$
* **Range:** $[0, \infty)$ — only non-negative values are produced

```python
import numpy as np

# In ML, the domain is the feature space
# A neural network might map R^784 -> R^10 (MNIST digits)
```

---

## 3. Types of Functions

### One-to-One (Injective)

A function is **one-to-one** if distinct inputs always produce distinct outputs:

$$
f(x_1) = f(x_2) \implies x_1 = x_2
$$

**ML Connection:** An injective encoding function ensures no two distinct inputs collapse to the same representation — critical for embedding layers in deep learning.

### Onto (Surjective)

A function is **onto** if every element in the codomain is the image of at least one element in the domain:

$$
\forall y \in B, \; \exists x \in A \text{ such that } f(x) = y
$$

**ML Connection:** A classification head that maps to the full probability simplex (every class has a non-zero probability path) is effectively surjective.

### One-to-One Correspondence (Bijective)

A function is **bijective** if it is both one-to-one and onto. This means there exists a perfect inverse $f^{-1}: B \to A$.

**ML Connection:** Flow-based generative models (like RealNVP or Glow) learn bijective transformations between a simple distribution and a complex data distribution, enabling exact density estimation.

### Identity Function

$$
f(x) = x
$$

Trivial, but essential — it is the neutral element under function composition and the basis of residual connections in deep neural networks.

### Exponential Function

$$
f(x) = e^x
$$

The foundation of the **Softmax function** used in virtually every classification network:

$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

```python
import numpy as np

def softmax(z):
    exp_z = np.exp(z - np.max(z))  # numerical stability trick
    return exp_z / exp_z.sum()

logits = np.array([2.0, 1.0, 0.1])
print(softmax(logits))  # [0.659, 0.242, 0.099]
```

---

## 4. Even and Odd Functions

### Even Functions

$$
f(-x) = f(x)
$$

Symmetric about the $y$-axis.

### Odd Functions

$$
f(-x) = -f(x)
$$

Symmetric about the origin.

**ML Connection:** Weight initialization strategies sometimes exploit symmetry properties. For example, initializing weights from a symmetric distribution (like $\mathcal{N}(0, \sigma^2)$) ensures the initial network has no systematic bias in any direction.

---

## 5. Periodic Functions

A function is **periodic** with period $T$ if:

$$
f(x + T) = f(x) \quad \forall x
$$

**ML Connection:** Positional encodings in Transformers use periodic functions (sines and cosines) to encode sequence positions, allowing the model to generalize to sequences longer than those seen during training:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

---

## 6. Bounded and Unbounded Functions

A function $f$ is **bounded** if there exists a constant $M$ such that:

$$
|f(x)| \le M \quad \forall x \in \text{Domain}
$$

* The **sigmoid function** $\sigma(x) = \frac{1}{1+e^{-x}}$ is bounded: $\sigma(x) \in (0, 1)$.
* The **ReLU function** $f(x) = \max(0, x)$ is unbounded above.

**Deep Learning Failure Mode:** Unbounded activation functions can cause activations to grow without limit during forward passes, leading to numerical overflow and exploding gradients. This is why Batch Normalization and careful weight initialization exist.

---

## 7. Composition of Functions

Given $f: A \to B$ and $g: B \to C$, the **composite function** $(g \circ f): A \to C$ is defined as:

$$
(g \circ f)(x) = g(f(x))
$$

**Deep Learning Connection:** A neural network with $L$ layers is literally a composition of $L$ functions:

$$
\hat{y} = (f_L \circ f_{L-1} \circ \dots \circ f_1)(\mathbf{x})
$$

Each layer $f_i(\mathbf{x}) = \sigma(W_i \mathbf{x} + \mathbf{b}_i)$ is a function, and the entire network is their composition. Backpropagation computes gradients through this composition using the **Chain Rule**.

```python
import torch.nn as nn

# A 3-layer network is a composition of 3 functions
network = nn.Sequential(
    nn.Linear(784, 128),   # f1
    nn.ReLU(),             # activation
    nn.Linear(128, 64),    # f2
    nn.ReLU(),             # activation
    nn.Linear(64, 10)      # f3
)

# Forward pass: y = f3(f2(f1(x)))
```

---

## 8. Inverse Functions

If $f: A \to B$ is bijective, the **inverse function** $f^{-1}: B \to A$ satisfies:

$$
f^{-1}(f(x)) = x \quad \text{and} \quad f(f^{-1}(y)) = y
$$

**ML Connection:** Invertible neural networks (used in normalizing flows) learn both a forward transformation and its exact inverse, enabling both generation and density estimation. The determinant of the Jacobian (Lecture 34) measures how the function stretches or compresses volume — essential for computing likelihoods.

---

## 9. Why Functions Matter in ML

| Concept | Function Role |
|:---|:---|
| **Model** | $f: \mathbb{R}^n \to \mathbb{R}^k$ (features to predictions) |
| **Loss Function** | $\mathcal{L}: \mathbb{R}^k \to \mathbb{R}$ (predictions to scalar error) |
| **Activation** | $\sigma: \mathbb{R} \to \mathbb{R}$ (introduces non-linearity) |
| **Optimizer** | Updates parameters to minimize $\mathcal{L}(f_\theta(\mathbf{x}), y)$ |

> **Check your intuition:** Why must the sigmoid function be bounded between 0 and 1? *(Answer: Because it is interpreted as a probability. If it were unbounded, the output could not be a valid probability, and the cross-entropy loss would produce nonsensical gradients.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 30: Minimal Polynomial and Jordan Canonical Form-II](Lecture%2030%20Minimal%20Polynomial%20and%20Jordan%20Canonical%20Form-II.md) — concludes the linear algebra foundation needed for understanding functions and their derivatives
- **Next:** [Lecture 32: Limits and Continuity](Lecture%2032%20Limits%20and%20Continuity.md) — extends function concepts to limits, which are essential for defining derivatives
- **Related:** [Lecture 33: Functions of N Variables](Lecture%2033%20Functions%20of%20N%20Variables.md) — generalizes single-variable functions to multivariate settings used in ML
- **Related:** [Lecture 34: Jacobian](Lecture%2034%20Jacobian.md) — describes how vector-valued functions change, building on the function framework
