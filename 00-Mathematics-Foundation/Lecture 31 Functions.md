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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $f(x_1) = f(x_2)$ | Equal output values | Assumes two inputs produce the same output |
| $\implies$ | Implication arrow | "Implies that" — logical consequence |
| $x_1 = x_2$ | Equal inputs | Conclusion that the two inputs must be the same |
| Injective (one-to-one) | Distinct inputs → distinct outputs | No two different inputs map to the same output; critical for encoding uniqueness |

**ML Connection:** An injective encoding function ensures no two distinct inputs collapse to the same representation — critical for embedding layers in deep learning.

### Onto (Surjective)

A function is **onto** if every element in the codomain is the image of at least one element in the domain:

$$
\forall y \in B, \; \exists x \in A \text{ such that } f(x) = y
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\forall y \in B$ | For every element $y$ in codomain $B$ | Universal quantifier — the condition must hold for ALL possible outputs |
| $\exists x \in A$ | There exists an element $x$ in domain $A$ | Existence quantifier — for each output there is at least one input |
| $\text{such that } f(x) = y$ | Mapping condition | Each codomain element is "hit" by some domain element |
| Surjective (onto) | Codomain = Range | Every possible output is actually achieved by some input |

**ML Connection:** A classification head that maps to the full probability simplex (every class has a non-zero probability path) is effectively surjective.

### One-to-One Correspondence (Bijective)

A function is **bijective** if it is both one-to-one and onto. This means there exists a perfect inverse $f^{-1}: B \to A$.

**ML Connection:** Flow-based generative models (like RealNVP or Glow) learn bijective transformations between a simple distribution and a complex data distribution, enabling exact density estimation.

### Identity Function

$$
f(x) = x
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $f(x)$ | Identity function | Returns its input unchanged |
| $x$ | Input variable | The domain element is passed through without transformation |
| Identity function | Neutral element of composition | Composing any function $g$ with identity gives $g$; basis of residual connections (ResNet) |

Trivial, but essential — it is the neutral element under function composition and the basis of residual connections in deep neural networks.

### Exponential Function

$$
f(x) = e^x
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $f(x)$ | Exponential function | A function where the independent variable $x$ appears in the exponent |
| $e$ | Euler's number ($\approx 2.718$) | The base of natural exponentials; has the unique property $\frac{d}{dx}e^x = e^x$ |
| $e^x$ | Exponential growth | Grows faster than any polynomial; used in softmax, sigmoid, and probability models |

The foundation of the **Softmax function** used in virtually every classification network:

$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\text{Softmax}(z_i)$ | Softmax output for class $i$ | Converts a vector of raw scores (logits) into a probability distribution |
| $z_i$ | Logit (raw score) for class $i$ | The unnormalized output of the last linear layer before classification |
| $e^{z_i}$ | Exponentiated logit | Maps any real number to a positive value; ensures all outputs are positive |
| $\sum_j e^{z_j}$ | Sum over all classes | Normalization factor guaranteeing the output sums to 1 |
| $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | Normalized probability | Produces a valid probability distribution over $K$ classes |

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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $f(-x)$ | Function evaluated at $-x$ | The function value when the input is negated |
| $f(x)$ | Function evaluated at $x$ | The original function value at the positive input |
| $f(-x) = f(x)$ | Even function condition | The graph is symmetric about the $y$-axis; only even powers in polynomial expansion |

Symmetric about the $y$-axis.

### Odd Functions

$$
f(-x) = -f(x)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $f(-x)$ | Function evaluated at $-x$ | Function value at the negated input |
| $-f(x)$ | Negated function value | The negative of the function value at the positive input |
| $f(-x) = -f(x)$ | Odd function condition | The graph is symmetric about the origin; only odd powers in polynomial expansion |

Symmetric about the origin.

**ML Connection:** Weight initialization strategies sometimes exploit symmetry properties. For example, initializing weights from a symmetric distribution (like $\mathcal{N}(0, \sigma^2)$) ensures the initial network has no systematic bias in any direction.

---

## 5. Periodic Functions

A function is **periodic** with period $T$ if:

$$
f(x + T) = f(x) \quad \forall x
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $f(x + T)$ | Function value shifted by period $T$ | The function at input $x + T$ |
| $f(x)$ | Original function value | The function at input $x$ |
| $T$ | Period | The smallest positive constant for which $f(x+T) = f(x)$ holds |
| $\forall x$ | For all $x$ | The equality must hold for every input in the domain |
| Periodic function | Repeating pattern | Function repeats its values at regular intervals of length $T$ |

**ML Connection:** Positional encodings in Transformers use periodic functions (sines and cosines) to encode sequence positions, allowing the model to generalize to sequences longer than those seen during training:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $PE_{(pos, 2i)}$ | Positional encoding at position $pos$, dimension $2i$ | Encodes the position of a token in a sequence for the Transformer model |
| $pos$ | Position index | The token's position in the sequence (0, 1, 2, ...) |
| $i$ | Dimension index | Which frequency band of the encoding ($0 \le i < d/2$) |
| $d$ | Model dimension | The total dimension of the embedding space |
| $\frac{pos}{10000^{2i/d}}$ | Frequency scaling | Higher dimensions get lower frequencies, creating a unique encoding per position |

---

## 6. Bounded and Unbounded Functions

A function $f$ is **bounded** if there exists a constant $M$ such that:

$$
|f(x)| \le M \quad \forall x \in \text{Domain}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $|f(x)|$ | Absolute value of $f(x)$ | Measures the magnitude of the function's output |
| $M$ | Bound constant | A finite number that caps the function's output magnitude |
| $\le M$ | Inequality condition | The function never exceeds this bound |
| $\forall x \in \text{Domain}$ | For all domain elements | The bound must hold for every valid input |
| Bounded function | Output stays within $[-M, M]$ | Outputs are confined to a finite range; essential for stable activations |

* The **sigmoid function** $\sigma(x) = \frac{1}{1+e^{-x}}$ is bounded: $\sigma(x) \in (0, 1)$.
* The **ReLU function** $f(x) = \max(0, x)$ is unbounded above.

**Deep Learning Failure Mode:** Unbounded activation functions can cause activations to grow without limit during forward passes, leading to numerical overflow and exploding gradients. This is why Batch Normalization and careful weight initialization exist.

---

## 7. Composition of Functions

Given $f: A \to B$ and $g: B \to C$, the **composite function** $(g \circ f): A \to C$ is defined as:

$$
(g \circ f)(x) = g(f(x))
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $(g \circ f)$ | Composite function $g$ composed with $f$ | Read as "$g$ after $f$" — apply $f$ first, then $g$ |
| $g(f(x))$ | Nested function evaluation | $f$ is evaluated at $x$, then $g$ is evaluated at $f(x)$ |
| $f: A \to B$ | First (inner) function | Maps domain $A$ to intermediate set $B$ |
| $g: B \to C$ | Second (outer) function | Maps intermediate set $B$ to final codomain $C$ |
| Function composition | Building complex functions from simple ones | The mathematical foundation of neural network layers — a network is one giant composition |

**Deep Learning Connection:** A neural network with $L$ layers is literally a composition of $L$ functions:

$$
\hat{y} = (f_L \circ f_{L-1} \circ \dots \circ f_1)(\mathbf{x})
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\hat{y}$ | Network output (prediction) | The final prediction after all $L$ layers |
| $f_i$ | $i$-th layer function | Each layer is a function: $f_i(\mathbf{x}) = \sigma(W_i\mathbf{x} + \mathbf{b}_i)$ |
| $\circ$ | Composition operator | Layers are applied sequentially; output of one is input to the next |
| $\mathbf{x}$ | Network input | The raw feature vector (e.g., pixel values) |
| $L$ | Number of layers | Network depth — deeper networks can learn more abstract features |

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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $f^{-1}$ | Inverse function | The function that "undoes" $f$, mapping each output back to its original input |
| $f^{-1}(f(x)) = x$ | Left inverse property | Applying $f$ then $f^{-1}$ recovers the original $x$ — the composition equals identity |
| $f(f^{-1}(y)) = y$ | Right inverse property | Applying $f^{-1}$ then $f$ recovers the original $y$ |
| Bijective $f$ | One-to-one and onto | Only bijective functions have a true inverse (both left and right) |
| Invertible | Bidirectional mapping | Normalizing flows learn invertible transformations for exact density estimation |

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
