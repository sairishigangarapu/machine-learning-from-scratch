# Multi-Output Networks and Loss Functions

## 1. Multiple Inputs and Outputs: The Weight Matrix

### Motivation and Intuition

Real-world classification problems rarely have a single output. Predicting which of 10 digits a handwritten image represents, or which of 100 species a plant belongs to, requires a network with multiple output nodes. Each input feature connects to every output through its own weight, and the entire transformation is a matrix multiplication.

Instead of individual weights $w_1, w_2, \dots$, we organize all weights into a matrix $\mathbf{W}$ where $w_{ji}$ connects input $i$ to output $j$.

### Formal Definition: Forward Pass in Matrix Form

For an input vector $\mathbf{x} \in \mathbb{R}^n$ and $m$ output classes, the raw outputs (logits) $\mathbf{z} \in \mathbb{R}^m$ are:

$$
\mathbf{z} = \mathbf{W} \mathbf{x} + \mathbf{b}
$$

In expanded form for $n=2$ inputs and $m=3$ outputs:

$$
\begin{bmatrix} z_1 \\ z_2 \\ z_3 \end{bmatrix} =
\begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \\ w_{31} & w_{32} \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} +
\begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix}
$$

Each logit $z_j$ is:

$$
z_j = \sum_{i=1}^{n} w_{ji} x_i + b_j
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $x_i$ | Input feature $i$ | Raw data fed into the network |
| $w_{ji}$ | Weight connecting input $i$ to output $j$ | Controls how much input $i$ influences output $j$ |
| $b_j$ | Bias for output node $j$ | Allows per-class offset |
| $z_j$ | Raw logit for class $j$ | Pre-activation value for output $j$ |
| $\mathbf{W} \in \mathbb{R}^{m \times n}$ | Weight matrix | All connections in a single compact object |
| $n$ | Number of input features | Dimensionality of the input |
| $m$ | Number of output classes | Number of categories to predict |

### Example: 3 Inputs, 2 Outputs

Suppose $\mathbf{x} = [0.5, -0.3, 0.8]^T$ and:

$$
\mathbf{W} = \begin{bmatrix} 0.2 & -0.5 & 0.3 \\ 0.1 & 0.4 & -0.2 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 0.1 \\ -0.1 \end{bmatrix}
$$

**Logits:**

$$
z_1 = 0.2(0.5) + (-0.5)(-0.3) + 0.3(0.8) + 0.1 = 0.1 + 0.15 + 0.24 + 0.1 = 0.59
$$

$$
z_2 = 0.1(0.5) + 0.4(-0.3) + (-0.2)(0.8) - 0.1 = 0.05 - 0.12 - 0.16 - 0.1 = -0.33
$$

These logits are then passed through an output activation function (softmax for multi-class classification) to produce probabilities.

> **ML Connection:** The weight matrix formulation is how neural network libraries (PyTorch, TensorFlow) implement fully connected layers. A `nn.Linear(n, m)` layer stores exactly this weight matrix and bias vector.

---

## 2. ArgMax vs SoftMax

### Motivation and Intuition

Raw logits $\mathbf{z}$ can be any real number — they are not probabilities. To make a decision, we need to convert logits into interpretable outputs. Two common approaches:

- **ArgMax:** Makes a hard decision. Picks the class with the largest logit and assigns it probability 1, all others 0.
- **SoftMax:** Makes a soft decision. Converts logits into a probability distribution that sums to 1, preserving relative order.

SoftMax is differentiable, which is essential for gradient-based training. ArgMax is not differentiable (its gradient is zero or undefined everywhere).

### ArgMax

$$
\text{ArgMax}(z_i) = \begin{cases} 1 & \text{if } z_i = \max(z_1, z_2, \dots, z_m) \\ 0 & \text{otherwise} \end{cases}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $z_i$ | Logit for class $i$ | Input to ArgMax |
| $\text{ArgMax}(z_i)$ | 1 if $z_i$ is largest, else 0 | A hard, discrete decision |
| $\max(z_1, \dots, z_m)$ | The maximum logit | The "winning" class gets all the probability |

**Problem:** ArgMax has zero gradient almost everywhere and an undefined gradient at the boundary. It cannot be used for backpropagation.

### SoftMax

$$
\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{m} e^{z_j}} \quad \text{for } i = 1, \dots, m
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $z_i$ | Raw logit for class $i$ | Input to SoftMax |
| $\sigma(z_i)$ | Predicted probability for class $i$ | Between 0 and 1; all $\sigma(z_i)$ sum to 1 |
| $e^{z_i}$ | Exponentiated logit | Ensures positivity; amplifies differences between logits |
| $\sum_{j=1}^{m} e^{z_j}$ | Normalization constant (partition function) | Ensures the outputs sum to 1 |

### Worked Example

Logits: $\mathbf{z} = [2.0, 1.0, 0.1]$

**Step 1 — Exponentiate:**

$$
e^{2.0} = 7.389, \quad e^{1.0} = 2.718, \quad e^{0.1} = 1.105
$$

**Step 2 — Sum:** $7.389 + 2.718 + 1.105 = 11.212$

**Step 3 — Normalize:**

$$
\sigma(z_1) = \frac{7.389}{11.212} = 0.659, \quad
\sigma(z_2) = \frac{2.718}{11.212} = 0.242, \quad
\sigma(z_3) = \frac{1.105}{11.212} = 0.099
$$

Class 1 has probability 65.9%, class 2 has 24.2%, class 3 has 9.9%. They sum to 1.

### SoftMax with Temperature

The temperature parameter $T$ controls the "sharpness" of the distribution:

$$
\sigma(z_i) = \frac{e^{z_i / T}}{\sum_{j=1}^{m} e^{z_j / T}}
$$

| $T$ | Effect | When to Use |
| :--- | :--- | :--- |
| $T = 1$ | Standard SoftMax | Default |
| $T > 1$ | Softer distribution (more uniform) | Knowledge distillation, exploration |
| $T < 1$ | Sharper distribution (more confident) | Low-temperature sampling for generative models |

> **ML Connection:** SoftMax is the standard output activation for multi-class classification. It is always paired with cross-entropy loss. The combination produces gradients $\hat{y} - y$ — the simplest possible error signal.

---

## 3. SoftMax Derivative: The Jacobian Matrix

### Motivation and Intuition

Because each SoftMax output $\sigma_i$ depends on ALL logits $z_k$ (through the denominator), the derivative has two cases:
- **Diagonal** ($i = k$): How does $\sigma_i$ change when its own logit $z_i$ changes?
- **Off-diagonal** ($i \neq k$): How does $\sigma_i$ change when a different logit $z_k$ changes?

The result is a full $m \times m$ Jacobian matrix.

### Diagonal Case ($i = k$)

Using the quotient rule:

$$
\frac{\partial \sigma_i}{\partial z_i} = \frac{e^{z_i} \cdot \sum_{j} e^{z_j} - e^{z_i} \cdot e^{z_i}}{\left(\sum_{j} e^{z_j}\right)^2}
= \frac{e^{z_i}}{\sum_{j} e^{z_j}} \cdot \frac{\sum_{j} e^{z_j} - e^{z_i}}{\sum_{j} e^{z_j}}
= \sigma_i (1 - \sigma_i)
$$

### Off-Diagonal Case ($i \neq k$)

$$
\frac{\partial \sigma_i}{\partial z_k} = \frac{0 \cdot \sum_{j} e^{z_j} - e^{z_i} \cdot e^{z_k}}{\left(\sum_{j} e^{z_j}\right)^2}
= -\frac{e^{z_i}}{\sum_{j} e^{z_j}} \cdot \frac{e^{z_k}}{\sum_{j} e^{z_j}}
= -\sigma_i \sigma_k
$$

### Full Jacobian

$$
J_{ik} = \frac{\partial \sigma_i}{\partial z_k} =
\begin{cases}
\sigma_i (1 - \sigma_i) & \text{if } i = k \\
-\sigma_i \sigma_k & \text{if } i \neq k
\end{cases}
$$

In matrix form:

$$
J = \text{diag}(\boldsymbol{\sigma}) - \boldsymbol{\sigma} \boldsymbol{\sigma}^T
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\sigma_i$ | SoftMax probability for class $i$ | Output of the SoftMax function |
| $z_k$ | Raw logit for class $k$ | Input to SoftMax |
| $\frac{\partial \sigma_i}{\partial z_i}$ | Diagonal Jacobian entry | Always positive; maximum at $\sigma_i = 0.5$ |
| $\frac{\partial \sigma_i}{\partial z_k}$ | Off-diagonal Jacobian entry | Always negative; increasing $z_k$ decreases $\sigma_i$ |
| $J \in \mathbb{R}^{m \times m}$ | Jacobian matrix | Full derivative of SoftMax w.r.t. all logits |

### Worked Example

From the earlier example: $\boldsymbol{\sigma} = [0.659, 0.242, 0.099]$

**Diagonal entries:**

$$
J_{11} = 0.659(1 - 0.659) = 0.225, \quad
J_{22} = 0.242(1 - 0.242) = 0.183, \quad
J_{33} = 0.099(1 - 0.099) = 0.089
$$

**Off-diagonal entries:**

$$
J_{12} = -0.659 \cdot 0.242 = -0.159, \quad
J_{13} = -0.659 \cdot 0.099 = -0.065
$$

$$
J_{21} = -0.242 \cdot 0.659 = -0.159, \quad
J_{23} = -0.242 \cdot 0.099 = -0.024
$$

$$
J_{31} = -0.099 \cdot 0.659 = -0.065, \quad
J_{32} = -0.099 \cdot 0.242 = -0.024
$$

Notice the symmetry: $J_{ik} = J_{ki}$ for $i \neq k$, but diagonals are different.

```python
import numpy as np

def softmax_jacobian(s):
    m = len(s)
    J = -np.outer(s, s)
    np.fill_diagonal(J, s * (1 - s))
    return J

s = np.array([0.659, 0.242, 0.099])
J = softmax_jacobian(s)
print(J.round(3))
```

> **ML Connection:** In practice, we rarely compute the full Jacobian. The combined cross-entropy + softmax gradient ($\hat{y} - y$) bypasses the Jacobian entirely. But understanding the Jacobian is essential for computing gradients when the loss is NOT cross-entropy or when softmax appears in intermediate layers.

---

## 4. Cross-Entropy Loss

### Motivation and Intuition

For multi-class classification, the output is a probability distribution over $K$ classes. We need a loss function that measures how different the predicted distribution $\hat{\mathbf{y}}$ is from the true distribution $\mathbf{y}$ (a one-hot vector). Cross-entropy is the standard choice.

Cross-entropy measures the number of additional bits needed to encode the true labels using the predicted distribution. Minimizing cross-entropy is equivalent to minimizing the KL divergence between the true and predicted distributions.

### Formula

For a single sample with $K$ classes:

$$
L = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)
$$

Since $\mathbf{y}$ is one-hot ($y_{\text{correct}} = 1$, all others $0$), this simplifies to:

$$
L = -\log(\hat{y}_{\text{correct}})
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $y_i$ | True label for class $i$ (0 or 1) | One-hot encoded ground truth |
| $\hat{y}_i$ | Predicted probability for class $i$ | SoftMax output (between 0 and 1) |
| $L$ | Cross-entropy loss | Scalar — how wrong the prediction is |
| $\log(\hat{y}_{\text{correct}})$ | Log-probability of the correct class | Drives toward $-\infty$ as $\hat{y}_{\text{correct}} \to 0$ (strong penalty) |

For a batch of $N$ samples:

$$
L = -\frac{1}{N} \sum_{j=1}^{N} \sum_{i=1}^{K} y_{ji} \log(\hat{y}_{ji})
$$

### Worked Example

True class: class 2 ($\mathbf{y} = [0, 1, 0]$), predictions: $\hat{\mathbf{y}} = [0.1, 0.8, 0.1]$

$$
L = -[0 \cdot \log(0.1) + 1 \cdot \log(0.8) + 0 \cdot \log(0.1)] = -\log(0.8) \approx 0.223
$$

If the prediction were less confident ($\hat{\mathbf{y}} = [0.3, 0.4, 0.3]$):

$$
L = -\log(0.4) \approx 0.916
$$

The loss is higher when the predicted probability for the correct class is lower.

### Why Cross-Entropy vs MSE for Classification

| Aspect | MSE | Cross-Entropy |
| :--- | :--- | :--- |
| Gradient w.r.t. logit $z$ | $2(\hat{y} - y) \cdot \hat{y}(1-\hat{y})$ (vanishes when $\hat{y}$ saturates) | $\hat{y} - y$ (always strong when prediction is wrong) |
| Probabilistic interpretation | Treats probabilities as real numbers | Measures distribution distance (KL divergence) |
| Convergence speed | Slow (vanishing gradients near 0 or 1) | Fast (linear error signal) |

Cross-entropy is strictly preferred for multi-class classification because its gradient does not vanish when predictions are near 0 or 1.

### Relationship to KL Divergence

$$
H(P, Q) = H(P) + D_{KL}(P \parallel Q)
$$

$H(P)$ is the entropy of the true distribution (constant for fixed one-hot labels). Minimizing cross-entropy is equivalent to minimizing KL divergence.

> **ML Connection:** Cross-entropy is the default loss for multi-class classification in every major deep learning framework. `nn.CrossEntropyLoss` in PyTorch combines softmax and cross-entropy into a single numerically stable operation.

---

## 5. The Combined Cross-Entropy + SoftMax Gradient

### Motivation and Intuition

In practice, softmax is almost always followed by cross-entropy loss. Computing the gradient of the loss with respect to the raw logits can be done in one step, bypassing the full Jacobian. The result is remarkably simple.

### Derivation

We want $\frac{\partial L}{\partial z_k}$ where:

$$
L = -\sum_{i=1}^{K} y_i \log(\hat{y}_i), \quad \hat{y}_i = \sigma(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Using the chain rule:

$$
\frac{\partial L}{\partial z_k} = \sum_{i=1}^{K} \frac{\partial L}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z_k}
$$

**Step 1 — Derivative of cross-entropy w.r.t. prediction:**

$$
\frac{\partial L}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i}
$$

**Step 2 — SoftMax Jacobian (compact form):**

$$
\frac{\partial \hat{y}_i}{\partial z_k} = \hat{y}_i (\delta_{ik} - \hat{y}_k), \quad \delta_{ik} = \begin{cases} 1 & \text{if } i = k \\ 0 & \text{if } i \neq k \end{cases}
$$

**Step 3 — Chain rule:**

$$
\frac{\partial L}{\partial z_k} = \sum_i -\frac{y_i}{\hat{y}_i} \cdot \hat{y}_i (\delta_{ik} - \hat{y}_k) = -\sum_i y_i (\delta_{ik} - \hat{y}_k)
$$

$$
= -\left( y_k - \hat{y}_k \sum_i y_i \right) = -\left( y_k - \hat{y}_k \cdot 1 \right)
$$

Since $\sum_i y_i = 1$ (one-hot labels sum to 1):

$$
\boxed{\frac{\partial L}{\partial z_k} = \hat{y}_k - y_k}
$$

In vector form:

$$
\boxed{\frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y}}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $z_k$ | Raw logit for class $k$ | Input to SoftMax |
| $y_k$ | True label for class $k$ (0 or 1) | Ground truth |
| $\hat{y}_k$ | Predicted probability for class $k$ | SoftMax output |
| $\frac{\partial L}{\partial z_k}$ | Gradient of loss w.r.t. logit $k$ | $\hat{y}_k - y_k$ — elegant and simple |

### Worked Example

True class is class 1 ($\mathbf{y} = [1, 0, 0]$), predictions $\hat{\mathbf{y}} = [0.8, 0.15, 0.05]$:

$$
\frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y} = [0.8 - 1, 0.15 - 0, 0.05 - 0] = [-0.2, 0.15, 0.05]
$$

**Interpretation:**
- Class 1: gradient is $-0.2$ (negative) — we need to INCREASE $z_1$ (because $\hat{y}_1 = 0.8$ is below the target $1.0$).
- Class 2: gradient is $+0.15$ (positive) — we need to DECREASE $z_2$ (because $\hat{y}_2 = 0.15$ is above the target $0.0$).
- Class 3: gradient is $+0.05$ (positive) — we need to DECREASE $z_3$ slightly.

```python
import numpy as np

def ce_softmax_gradient(y_true, y_pred):
    return y_pred - y_true

y = np.array([1, 0, 0])
y_hat = np.array([0.8, 0.15, 0.05])
grad = ce_softmax_gradient(y, y_hat)
print(f"Gradient: {grad}")  # [-0.2, 0.15, 0.05]
```

> **ML Connection:** This combined gradient is why cross-entropy and softmax are paired. The gradient $\hat{y} - y$ never vanishes (unless the prediction is perfect), providing a strong error signal at all times. Compare this to MSE + sigmoid, where the gradient includes $\sigma'(z)$ which vanishes near saturation.

---

## 6. Full Backpropagation with Cross-Entropy and Softmax

### Motivation and Intuition

With the combined CE + SoftMax gradient $\delta_{\text{out}} = \hat{\mathbf{y}} - \mathbf{y}$, backpropagation through the rest of the network proceeds exactly as before. The error signal at the output is simply the difference between predictions and targets.

### The Network (Multi-Class)

Consider a network with one hidden layer (ReLU) and a softmax output layer for $K$-class classification:

**Forward pass:**

$$
\mathbf{h} = \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \in \mathbb{R}^{d_h}
$$

$$
\mathbf{z} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2 \in \mathbb{R}^{K}
$$

$$
\hat{\mathbf{y}} = \text{softmax}(\mathbf{z})
$$

**Loss (cross-entropy for a single sample):**

$$
L = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)
$$

### Backward Pass

**Step 1 — Output error signal:**

$$
\frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y} \quad \text{(vector of size $K$)}
$$

**Step 2 — Output layer gradients (weight matrix $\mathbf{W}_2$ and bias $\mathbf{b}_2$):**

For each output node $j$ and hidden node $i$:

$$
\frac{\partial L}{\partial w_{ji}^{(2)}} = \frac{\partial L}{\partial z_j} \cdot h_i = (\hat{y}_j - y_j) \cdot h_i
$$

$$
\frac{\partial L}{\partial b_j^{(2)}} = \frac{\partial L}{\partial z_j} = \hat{y}_j - y_j
$$

In matrix form:

$$
\frac{\partial L}{\partial \mathbf{W}_2} = (\hat{\mathbf{y}} - \mathbf{y}) \mathbf{h}^T \in \mathbb{R}^{K \times d_h}
$$

$$
\frac{\partial L}{\partial \mathbf{b}_2} = \hat{\mathbf{y}} - \mathbf{y} \in \mathbb{R}^{K}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\frac{\partial L}{\partial \mathbf{z}}$ | Error signal at output | $\hat{\mathbf{y}} - \mathbf{y}$ — the elegant combined gradient |
| $\frac{\partial L}{\partial w_{ji}^{(2)}}$ | Gradient for output weight connecting hidden $i$ to output $j$ | Error signal $\times$ hidden activation |
| $\frac{\partial L}{\partial \mathbf{W}_2}$ | Gradient for entire output weight matrix | Outer product of error signal and hidden vector |

**Step 3 — Backpropagate to hidden layer:**

The error signal for the hidden layer is obtained by propagating the output error through $\mathbf{W}_2$ and applying the ReLU derivative:

$$
\frac{\partial L}{\partial \mathbf{h}} = \mathbf{W}_2^T (\hat{\mathbf{y}} - \mathbf{y}) \in \mathbb{R}^{d_h}
$$

$$
\frac{\partial L}{\partial \mathbf{z}_1} = \frac{\partial L}{\partial \mathbf{h}} \odot \text{ReLU}'(\mathbf{z}_1) \quad \text{(element-wise)}
$$

where $\mathbf{z}_1 = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$ is the hidden pre-activation.

**Step 4 — Hidden layer gradients:**

$$
\frac{\partial L}{\partial \mathbf{W}_1} = \frac{\partial L}{\partial \mathbf{z}_1} \mathbf{x}^T \in \mathbb{R}^{d_h \times n}
$$

$$
\frac{\partial L}{\partial \mathbf{b}_1} = \frac{\partial L}{\partial \mathbf{z}_1} \in \mathbb{R}^{d_h}
$$

### Complete Algorithm Summary

```
Forward pass:
  h = ReLU(W1 @ x + b1)       # hidden activations
  z = W2 @ h + b2              # output logits
  y_hat = softmax(z)           # probabilities
  L = cross_entropy(y, y_hat)  # loss

Backward pass (gradients):
  dz = y_hat - y                             # output error signal
  dW2 = outer(dz, h)                         # = dz @ h^T
  db2 = dz                                   # bias gradient
  dh = W2^T @ dz                             # backprop to hidden
  dz1 = dh * relu'(z1)                       # through activation
  dW1 = outer(dz1, x)                        # = dz1 @ x^T
  db1 = dz1                                  # bias gradient

Update:
  W1 -= alpha * dW1;  b1 -= alpha * db1
  W2 -= alpha * dW2;  b2 -= alpha * db2
```

> **ML Connection:** This is exactly what `backward()` computes in PyTorch's `CrossEntropyLoss` combined with `Linear` layers. The modularity of backpropagation means the CE + SoftMax combination produces a clean output gradient that flows naturally through any preceding layers.

---

### Python: SoftMax, Cross-Entropy, and Combined Gradient from Scratch

```python
import numpy as np

def softmax(z):
    z_shifted = z - np.max(z, axis=-1, keepdims=True)  # numerical stability
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-15))  # 1e-15 for numerical stability

def ce_softmax_gradient(y_true, y_pred):
    return y_pred - y_true

# Demo: 3-class classification
y = np.array([0, 1, 0])  # true class is class 2
z = np.array([1.5, 0.5, -0.5])  # raw logits

y_hat = softmax(z)
loss = cross_entropy(y, y_hat)
grad = ce_softmax_gradient(y, y_hat)

print(f"Logits:     {z}")
print(f"Softmax:    {y_hat}")
print(f"Loss:       {loss:.4f}")
print(f"Gradient:   {grad}")
```

---

> **Check your intuition:** In a 3-class problem, the true label is class 3 ($\mathbf{y} = [0, 0, 1]$). The logits before softmax are $\mathbf{z} = [1.0, 2.0, 3.0]$. Compute the softmax probabilities and the gradient $\partial L / \partial \mathbf{z}$.

<details>
<summary>Answer</summary>
$e^1 = 2.718, e^2 = 7.389, e^3 = 20.086$. Sum = 30.193.
$\hat{y} = [2.718/30.193, 7.389/30.193, 20.086/30.193] = [0.090, 0.245, 0.665]$.
$\partial L / \partial \mathbf{z} = [0.090 - 0, 0.245 - 0, 0.665 - 1] = [0.090, 0.245, -0.335]$.
The gradient is positive for classes 1 and 2 (we must decrease their logits) and negative for class 3 (we must increase its logit).
</details>

---

## Prerequisites and Further Reading

- **Previous:** 01-Neural-Network-Fundamentals (forward pass, backpropagation), 02-Activation-Functions
- **Next:** 04-Convolutional-Neural-Networks (extending to spatial data)
- **Related:** L09-L13 StatQuest (multiple inputs/outputs, softmax, cross-entropy, backprop)
- **Foundational:** Matrix multiplication (weight matrix form), partial derivatives (Jacobians), probability (distributions, KL divergence) from 00-Mathematics-Foundation
- **Further:** "Pattern Recognition and Machine Learning" (Bishop, 2006) Chapter 4 — probabilistic generative models and softmax
