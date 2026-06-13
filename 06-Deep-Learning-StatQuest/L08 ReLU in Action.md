## L08 ReLU in Action

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. The Problem with Sigmoid

### Motivation and Intuition

In L02-L07, we used the sigmoid activation function in the hidden layer. Sigmoid works, but it has a major flaw for deep networks: it saturates. When the input to sigmoid is very positive or very negative, the output is close to 1 or 0, and the derivative is nearly zero. Remember from backpropagation that the gradient through a sigmoid node includes the term $\sigma(z)(1 - \sigma(z))$. If this term is near zero, the gradient vanishes, and the weight stops learning.

This is called the **vanishing gradient problem**. In deep networks with many layers, the gradients shrink exponentially as they propagate backward. The first layers learn excruciatingly slowly or not at all. The ReLU activation function solves this problem.

### The Vanishing Gradient in Action

In the sigmoid-based network from L07, the gradient for $w_1$ included:

$$
\frac{\partial L}{\partial w_1} = \delta_3 \cdot w_3 \cdot \sigma(z_1)(1 - \sigma(z_1)) \cdot x
$$

If $\sigma(z_1)$ is near 0 or 1, then $\sigma(z_1)(1 - \sigma(z_1)) \approx 0$, and the gradient for $w_1$ vanishes.

---

## 2. The ReLU Activation Function

### What ReLU Is

ReLU (Rectified Linear Unit) is the simplest non-linear activation function:

$$
\text{ReLU}(z) = \max(0, z)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\text{ReLU}(z)$ | ReLU output | $z$ if $z > 0$, otherwise 0 — a simple threshold at zero |
| $z$ | Pre-activation (weighted sum + bias) | The input to the activation function |
| $\max(0, z)$ | Maximum operation | If the input is positive, pass it through unchanged. If negative, output 0 |

### The Derivative of ReLU

$$
\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \end{cases}
$$

(At $z = 0$, the derivative is technically undefined, but in practice we set it to 0 or 1.)

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\text{ReLU}'(z)$ | Derivative of ReLU w.r.t. $z$ | 1 when $z > 0$, 0 when $z < 0$ — no vanishing gradient for active neurons! |
| 1 (for $z > 0$) | Gradient for active neurons | The gradient passes through unchanged — no attenuation |
| 0 (for $z < 0$) | Gradient for inactive neurons | The neuron does not fire, its gradient is zero, and it stops learning |

### Why ReLU Fixes the Vanishing Gradient

With ReLU, the gradient through an active neuron ($z > 0$) is always 1 — no squashing, no saturation. Compare:

- Sigmoid derivative: $\sigma(z)(1 - \sigma(z)) \leq 0.25$
- ReLU derivative: $1$ (for $z > 0$)

In deep networks, multiplying by 1 at each layer preserves the gradient magnitude. Multiplying by at most 0.25 at each layer shrinks it exponentially. A 10-layer sigmoid network can shrink the gradient by $0.25^{10} \approx 10^{-6}$. With ReLU, active neurons pass the full gradient.

---

## 3. ReLU in a Neural Network

### Switching from Sigmoid to ReLU

If we swap the sigmoid in the hidden layer for ReLU in our example network:

$$
z_1 = w_1 x + b_1, \quad h_1 = \text{ReLU}(z_1)
$$

$$
z_2 = w_2 x + b_2, \quad h_2 = \text{ReLU}(z_2)
$$

$$
z_3 = w_3 h_1 + w_4 h_2 + b_3, \quad \hat{y} = \sigma(z_3) \text{ (output still needs sigmoid for binary classification)}
$$

### The Effect on the Squiggle

With sigmoid hidden nodes, each node contributes a smooth S-shaped bump. With ReLU hidden nodes, each node contributes a sharp "hinge" — zero on one side, linear on the other. Adding two ReLU hinges produces a tent-like shape. Adding more ReLU nodes produces a piecewise linear squiggle.

ReLU networks produce **piecewise linear** decision boundaries. The non-linearity comes from the sharp bend at zero, not from a smooth curve. This is actually sufficient to approximate any function, given enough hidden nodes.

### Python Example — ReLU vs Sigmoid

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compare derivatives
z_vals = np.array([-5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0])
print("z      sigmoid'   relu'")
for z in z_vals:
    sig_deriv = sigmoid(z) * (1 - sigmoid(z))
    relu_deriv = 1.0 if z > 0 else 0.0
    print(f"{z:+.1f}   {sig_deriv:.4f}     {relu_deriv:.1f}")

# Tiny ReLU network forward pass
def forward_relu(x, w1, b1, w2, b2, w3, b3, w4, b4):
    z1 = w1 * x + b1
    h1 = relu(z1)
    z2 = w2 * x + b2
    h2 = relu(z2)
    z3 = w3 * h1 + w4 * h2 + b3
    y_pred = sigmoid(z3)
    return y_pred, h1, h2

# Same parameters as before but with ReLU
dosage = 0.5
w1, b1 = 1.70, -0.85
w2, b2 = -1.70, 0.85
w3, b3, w4 = 1.00, 0.00, -1.00

pred, h1, h2 = forward_relu(dosage, w1, b1, w2, b2, w3, b3, w4, b4)
print(f"\nReLU network prediction: {pred:.4f}")
print(f"Hidden node 1 (ReLU): {h1:.4f}, Hidden node 2 (ReLU): {h2:.4f}")
```

---

## 4. The Dying ReLU Problem

### What It Is

When a ReLU neuron gets stuck with $z < 0$ for all training samples, its output is always 0, and its gradient is always 0. Once this happens, the neuron never recovers — it is "dead." This is called the **dying ReLU problem**.

### Why It Happens

A large gradient update can push the bias far negative. If the weighted sum $z = wx + b$ is negative for every input, the ReLU output is 0 and the gradient is 0. The neuron makes no further updates and remains dead permanently.

### How to Detect It

During training, track the fraction of ReLU neurons that produce positive outputs. If many neurons rarely or never fire (output > 0), they may be dead.

---

## 5. ReLU Variants

### Leaky ReLU

Leaky ReLU allows a small non-zero gradient when $z < 0$:

$$
\text{LeakyReLU}(z) = \max(\alpha z, z)
$$

Typically $\alpha = 0.01$. The derivative is $1$ for $z > 0$ and $\alpha$ for $z < 0$.

| Variant | Formula | Gradient ($z > 0$) | Gradient ($z < 0$) | Solves Dying ReLU? |
| :--- | :--- | :--- | :--- | :--- |
| **ReLU** | $\max(0, z)$ | 1 | 0 | No |
| **Leaky ReLU** | $\max(\alpha z, z)$ | 1 | $\alpha$ (e.g., 0.01) | Yes — small gradient keeps neurons alive |
| **ELU** | $z$ if $z > 0$, $\alpha(e^z - 1)$ if $z \leq 0$ | 1 | $\alpha e^z$ | Yes — smooth negative tail |
| **Parametric ReLU** | $\max(\alpha z, z)$ where $\alpha$ is learned | 1 | $\alpha$ (learned) | Yes — adapts per neuron |

### ELU (Exponential Linear Unit)

ELU smooths the negative region:

$$
\text{ELU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha(e^z - 1) & \text{if } z \leq 0 \end{cases}
$$

The exponential tail keeps the mean activation closer to zero, which can speed up learning.

---

## 6. When to Use Which Activation

| Activation | Best For | Why |
| :--- | :--- | :--- |
| **ReLU** | Hidden layers (default) | Simple, fast, no vanishing gradient for active neurons |
| **Leaky ReLU** | Deep networks / when dead neurons are observed | The small negative gradient keeps more neurons alive |
| **ELU** | Networks where zero-mean activations help | Smoother gradient flow in the negative region |
| **Sigmoid** | Output layer for binary classification | Output bounded to (0, 1) — interpretable as probability |
| **Tanh** | Hidden layers (legacy) | Zero-centered, but still suffers from vanishing gradient |

For modern deep learning, ReLU is the default choice for hidden layers. Start with ReLU, switch to Leaky ReLU only if you observe dying neurons.

---

> **Check your intuition:** A deep network with sigmoid activations has 20 hidden layers. Roughly how much does the gradient shrink from the output layer to the first hidden layer (assuming average sigmoid derivative of 0.2)? How does this compare to a ReLU network where half the neurons are active?

---

## Prerequisites and Further Reading

- **Previous:** L02 Neural Networks Part 1 (activation functions introduced), L07 Backpropagation Details Part 2 (why gradients vanish)
- **Next:** L10 ArgMax and SoftMax (output activations for multi-class classification)
- **Related:** Vanishing Gradient Problem, Deep Learning fundamentals
