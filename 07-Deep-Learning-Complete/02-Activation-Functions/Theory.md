# Activation Functions: Non-Linearity in Neural Networks

## 1. Why Activation Functions?

### Motivation and Intuition

Without activation functions, a neural network is just a series of linear transformations. Stacking linear layers is equivalent to a single linear layer — no more powerful than linear regression or logistic regression. Activation functions introduce non-linearity, allowing the network to bend, curve, and twist the decision boundary to fit complex patterns.

Think of activation functions as the "bend" in a wire. A straight wire can only make straight shapes. Apply a bending force at each layer, and the wire can form any curve. Each activation function applies a controlled bend to the data as it flows through the network.

### The Linear Collapse Problem

Consider two linear layers:

$$
\mathbf{h} = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1, \quad \hat{y} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2
$$

This is equivalent to a single linear layer:

$$
\hat{y} = \mathbf{W}_2 (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (\mathbf{W}_2 \mathbf{W}_1) \mathbf{x} + (\mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2) = \mathbf{W}_{\text{eff}} \mathbf{x} + \mathbf{b}_{\text{eff}}
$$

No matter how many linear layers you stack, you never escape the set of linear functions. An activation function $\sigma$ breaks this collapse:

$$
\mathbf{h} = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \quad \hat{y} = \sigma(\mathbf{W}_2 \mathbf{h} + \mathbf{b}_2)
$$

Now the composition is genuinely non-linear and can represent arbitrarily complex functions (Universal Approximation Theorem).

| Concept | Without Activation | With Activation |
| :--- | :--- | :--- |
| Stacking layers | Collapses to one linear layer | Each layer adds expressivity |
| Decision boundary | Always linear (a hyperplane) | Can be arbitrarily curved |
| Function class | Linear functions only | All continuous functions (with enough neurons) |
| Example: XOR | Impossible | Trivially solvable |

> **ML Connection:** The choice of activation function is one of the first architectural decisions when designing a neural network. The wrong choice can prevent learning entirely (vanishing gradients) or slow it down dramatically.

---

## 2. Sigmoid Activation

### Motivation and Intuition

The sigmoid function was the default activation in early neural networks. It squashes any real number into the range $(0, 1)$, making it intuitive as a "firing rate" (0 = off, 1 = fully on). It is still used in the output layer for binary classification, but rarely in hidden layers of deep networks.

### Formula and Derivative

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\sigma(x)$ | Sigmoid output | Always between 0 and 1 — S-shaped curve |
| $e^{-x}$ | Exponential decay | When $x$ is large positive, $e^{-x} \to 0$, $\sigma(x) \to 1$; when $x$ is large negative, $\sigma(x) \to 0$ |
| $\sigma'(x)$ | Sigmoid derivative | Maximum of $0.25$ at $x=0$; approaches $0$ as $|x|$ grows |

### Vanishing Gradient Problem

The sigmoid derivative has a maximum value of $0.25$ and approaches $0$ for large positive or negative $x$. In deep networks, backpropagation multiplies these small derivatives across layers. After 10 layers with sigmoid, the gradient can shrink by $0.25^{10} \approx 10^{-6}$ — vanishingly small.

$$
\frac{\partial L}{\partial w_1} = \delta_{\text{out}} \cdot \prod_{l=1}^{L-1} \big( w_{l+1} \cdot \sigma'(z_l) \big) \cdot x
$$

If each $\sigma'(z_l) \leq 0.25$, the product of many such terms vanishes exponentially with depth.

| Problem | Cause | Effect |
| :--- | :--- | :--- |
| Vanishing gradient | Sigmoid derivative $\leq 0.25$; saturates near 0 and 1 | Early layers learn very slowly or not at all |
| Not zero-centered | Output always positive ($0$ to $1$) | Gradients for weights feeding into sigmoid are all positive or all negative, causing zigzagging optimization |

### When to Use Sigmoid

- Output layer for binary classification (probability output).
- NOT recommended for hidden layers in deep networks.

---

## 3. Tanh Activation

### Motivation and Intuition

Tanh (hyperbolic tangent) is a scaled and shifted version of sigmoid. It is zero-centered (outputs range $(-1, 1)$), which helps with optimization because gradients can be both positive and negative. However, it still saturates and suffers from vanishing gradients.

### Formula and Derivative

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1
$$

$$
\tanh'(x) = 1 - \tanh^2(x)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\tanh(x)$ | Tanh output | Between $-1$ and $1$ — zero-centered S-curve |
| $\tanh'(x)$ | Tanh derivative | Maximum of $1$ at $x=0$; still saturates to $0$ for large $|x|$ |
| $2\sigma(2x) - 1$ | Relationship to sigmoid | Tanh is just a shifted/scaled sigmoid |

### Comparison to Sigmoid

| Property | Sigmoid | Tanh |
| :--- | :--- | :--- |
| Range | $(0, 1)$ | $(-1, 1)$ |
| Zero-centered | No | Yes |
| Max derivative | $0.25$ | $1.0$ |
| Vanishing gradient | Yes | Yes (saturation) |

Tanh is strictly better than sigmoid for hidden layers because it is zero-centered, which reduces the zigzagging problem. However, it still saturates, so for very deep networks, ReLU and its variants are preferred.

### When to Use Tanh

- Hidden layers in shallower networks (1-3 hidden layers).
- Legacy applications; modern practice prefers ReLU.
- Certain sequence models (LSTM gates use tanh).

---

## 4. ReLU (Rectified Linear Unit)

### Motivation and Intuition

ReLU is the simplest non-linear activation: it passes positive values through unchanged and sets negative values to zero. Despite its simplicity, ReLU solved the vanishing gradient problem for deep networks and became the default activation for hidden layers.

The key insight: for any positive input, the derivative of ReLU is exactly 1. This means gradients flow backward through active neurons without attenuation — no vanishing gradient.

### Formula and Derivative

$$
f(x) = \max(0, x)
$$

$$
f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases}
$$

(At $x = 0$, the derivative is undefined; in practice we define it as 0 or 1.)

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $f(x)$ | ReLU output | $x$ if $x > 0$, otherwise $0$ |
| $f'(x)$ | ReLU derivative | $1$ for active neurons — gradient passes through unchanged |
| $\max(0, x)$ | Threshold at zero | Simple, computationally cheap (just a comparison) |

### Why ReLU Fixes Vanishing Gradients

Compare the derivative of sigmoid and ReLU through a deep network:

- Sigmoid: multiply by $\leq 0.25$ at each layer. After 20 layers: $\leq 0.25^{20} \approx 10^{-12}$.
- ReLU (active neurons): multiply by $1$ at each layer. After 20 layers: $1^{20} = 1$.

With ReLU, active neurons pass the full gradient backward. Deep networks (50+ layers) became trainable for the first time.

### The Dying ReLU Problem

When a ReLU neuron gets stuck with $x < 0$ for all training samples:

- Output is always $0$.
- Gradient is always $0$ (since $f'(x) = 0$ for $x < 0$).
- The neuron never recovers — it is permanently "dead."

**Causes:** A large gradient update pushes the bias far negative. If the weighted sum $z = wx + b$ is negative for every input, the ReLU dies.

**Detection:** Track the fraction of ReLU neurons that produce positive outputs during training. If many are always zero, they are dead.

### ReLU Forward and Backward Pass

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# Forward: h = relu(w * x + b)
# Backward: dL_dz = dL_dh * (1 if z > 0 else 0)
```

> **ML Connection:** ReLU is the default activation for hidden layers in virtually all modern neural networks: CNNs, RNNs, Transformers, and MLPs. Its simplicity and gradient-preserving property make deep learning practical.

---

## 5. ReLU Variants and Alternatives

### Motivation and Intuition

The dying ReLU problem motivated several variants that allow a small gradient when $x < 0$, keeping neurons alive. Each variant modifies the negative region while keeping the positive region as $f(x) = x$.

### Leaky ReLU

$$
f(x) = \max(\alpha x, x), \quad f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ \alpha & \text{if } x < 0 \end{cases}
$$

Typically $\alpha = 0.01$. The small negative gradient prevents neurons from dying entirely.

### Parametric ReLU (PReLU)

Same as Leaky ReLU, but $\alpha$ is a learnable parameter — each neuron learns its own negative slope.

### Exponential Linear Unit (ELU)

$$
f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}
$$

ELU smooths the negative region with an exponential tail. The mean activation is closer to zero, which can speed up learning.

### GELU (Gaussian Error Linear Unit)

$$
f(x) = x \cdot \Phi(x)
$$

where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution. GELU weights inputs by their value relative to the distribution of other inputs. It is used in Transformer models (BERT, GPT).

| Activation | Formula | Derivative ($x>0$) | Derivative ($x<0$) | Solves Dying ReLU? |
| :--- | :--- | :--- | :--- | :--- |
| ReLU | $\max(0, x)$ | 1 | 0 | No |
| Leaky ReLU | $\max(\alpha x, x)$ | 1 | $\alpha$ (0.01) | Yes |
| PReLU | $\max(\alpha x, x)$ | 1 | $\alpha$ (learned) | Yes |
| ELU | $x$ if $x>0$, $\alpha(e^x-1)$ else | 1 | $\alpha e^x$ | Yes |
| GELU | $x\Phi(x)$ | $\Phi(x) + x\phi(x)$ | smooth | Yes |

```python
import numpy as np

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))

def gelu(z):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))
```

> **ML Connection:** In practice, start with ReLU. If you observe dead neurons (check activation statistics during training), switch to Leaky ReLU. GELU is the standard in Transformer architectures. ELU is less common but useful when zero-mean activations are beneficial.

---

## 6. Activation Functions Comparison Table

| Activation | Formula | Range | Derivative Range | Vanishing Gradient? | Zero-Centered? | Comput. Cost | Default For |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear** | $x$ | $(-\infty, \infty)$ | $1$ | No | Yes | Very low | Output: regression |
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | $(0, 1)$ | $(0, 0.25]$ | Yes | No | Low | Output: binary classif. |
| **Tanh** | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | $(-1, 1)$ | $(0, 1]$ | Yes | Yes | Low | Hidden (shallow nets) |
| **ReLU** | $\max(0, x)$ | $[0, \infty)$ | $\{0, 1\}$ | No (when active) | No | Very low | **Hidden (default)** |
| **Leaky ReLU** | $\max(\alpha x, x)$ | $(-\infty, \infty)$ | $\{\alpha, 1\}$ | No | No | Very low | Hidden (dying ReLU) |
| **ELU** | $x$ if $x>0$, $\alpha(e^x-1)$ else | $(-\alpha, \infty)$ | $(0, 1]$ | No | Near-zero | Low | Hidden (zero-mean) |
| **GELU** | $x\Phi(x)$ | $(-\infty, \infty)$ | $(0, 1]$ | No | Near-zero | Medium | Hidden (Transformers) |
| **Swish** | $x\sigma(x)$ | $(-\infty, \infty)$ | varies | No | No | Medium | Hidden (NAS-found) |
| **Softmax** | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | $(0, 1)$, sum=1 | Jacobian | N/A | N/A | Medium | Output: multi-class |

### Key Takeaways

1. **Hidden layers:** Use ReLU as default. Switch to Leaky ReLU if dying neurons appear. Use GELU for Transformers.
2. **Output layer:** Linear for regression, sigmoid for binary classification, softmax for multi-class classification.
3. **Avoid** sigmoid and tanh in hidden layers of deep networks — they cause vanishing gradients.
4. **No free lunch:** More complex activations (GELU, Swish) may provide small improvements but increase compute cost.

---

## 7. Choosing the Right Activation Function

### Motivation and Intuition

The choice of activation function depends on where in the network it is used (hidden vs output), the task (regression, binary classification, multi-class), and the network depth. There is no single best activation — each has trade-offs.

### Output Layer Selection

| Task | Output Activation | Why |
| :--- | :--- | :--- |
| Regression (unbounded) | Linear ($f(x)=x$) | Output can be any real number |
| Regression (positive only) | ReLU or softplus | Constrains output to positive values |
| Binary classification | Sigmoid | Output in $(0,1)$ — interpretable as probability |
| Multi-class classification | Softmax | Outputs sum to 1 — a valid probability distribution |
| Multi-label classification | Sigmoid (per class) | Each class independently in $(0,1)$ |

### Hidden Layer Selection Guide

| Network Depth | Recommended Activation | Rationale |
| :--- | :--- | :--- |
| Shallow (1-2 hidden layers) | ReLU, tanh, or sigmoid | Vanishing gradient is not severe |
| Moderate (3-10 layers) | ReLU (default) | Avoids vanishing gradient, fast |
| Deep (10+ layers) | ReLU or Leaky ReLU | Gradient preservation is critical |
| Very deep (50+, ResNets) | ReLU + batch normalization | Skip connections + ReLU enable training |
| Transformers | GELU | Smooth, near-zero mean, empirically better |

### Practical Rules of Thumb

```python
def recommend_activation(task, depth):
    if task == "hidden":
        if depth > 10:
            return "ReLU or Leaky ReLU"
        elif depth > 3:
            return "ReLU"
        else:
            return "ReLU or tanh"
    elif task == "binary_output":
        return "sigmoid"
    elif task == "multiclass_output":
        return "softmax"
    elif task == "regression_output":
        return "linear"
```

> **ML Connection:** Activation function choice is a hyperparameter you should experiment with. Start with ReLU for hidden layers, then try Leaky ReLU or GELU if performance plateaus. Monitor activation statistics (fraction of dead ReLU units, mean activation values) to diagnose issues.

---

> **Check your intuition:** A deep network has 20 hidden layers using sigmoid activation. The average sigmoid derivative during training is $0.2$. By what factor is the gradient for the first hidden layer multiplied compared to the output layer? How would a ReLU network with 60% active neurons compare?

<details>
<summary>Answer</summary>
Sigmoid: gradient shrinks by approximately $0.2^{19} \approx 5 \times 10^{-14}$ — essentially zero. The first layer learns nothing. ReLU with 60% active: each layer multiplies by at most $0.6 \times 1 + 0.4 \times 0 = 0.6$ (some paths). The gradient shrinks by $0.6^{19} \approx 6 \times 10^{-5}$ — much better, and with residual connections it stays near 1.
</details>

---

## Prerequisites and Further Reading

- **Previous:** 01-Neural-Network-Fundamentals (forward pass, backpropagation, gradient descent)
- **Next:** 03-Multi-Output-and-Loss-Functions (softmax, cross-entropy for multi-class)
- **Related:** Vanishing Gradient Problem, Batch Normalization (mitigates saturation), Weight Initialization (affects activation behavior)
- **Foundational:** Calculus (derivatives, chain rule) from 00-Mathematics-Foundation
- **Further:** "Rectified Linear Units Improve Restricted Boltzmann Machines" (Nair & Hinton, 2010), "GELU: Gaussian Error Linear Units" (Hendrycks & Gimpel, 2016)
