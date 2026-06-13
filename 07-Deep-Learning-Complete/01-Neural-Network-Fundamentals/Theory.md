# Neural Network Fundamentals: From Perceptron to Backpropagation

## 1. The Perceptron and Neural Network Architecture

### Motivation and Intuition

A neural network is a curve-fitting machine. It takes inputs, transforms them through a series of weighted connections and non-linearities, and produces an output. The term "neural network" sounds biologically inspired, but in practice it is just a stack of mathematical operations that bend and stretch a function until it matches the data. There is no black magic — only weights, biases, and activation functions organized into layers.

The simplest building block is the **perceptron**: a single neuron that computes a weighted sum of its inputs, adds a bias, and passes the result through a threshold (or, in modern networks, a smooth activation function). A single perceptron can learn linearly separable patterns. To learn non-linear patterns (like XOR), we stack perceptrons into layers — this is the **Multilayer Perceptron (MLP)**.

### Architecture Overview

An MLP has three types of layers:

- **Input Layer** — One node per feature. No computation; just passes data forward.
- **Hidden Layers** — One or more layers between input and output. Each node applies a non-linear activation to a weighted sum of the previous layer's outputs.
- **Output Layer** — Produces the final prediction. The activation function depends on the task (sigmoid for binary classification, softmax for multi-class, linear for regression).

Every connection between nodes carries a **weight** that controls influence. Every node (except input nodes) has a **bias** that shifts the activation function left or right.

### Formal Definition: The Perceptron

A perceptron computes:

$$
y = \begin{cases} 1 & \text{if } \mathbf{w}^T\mathbf{x} + b \geq 0 \\ 0 & \text{otherwise} \end{cases}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{x} \in \mathbb{R}^n$ | Input feature vector | The raw data point being classified |
| $\mathbf{w} \in \mathbb{R}^n$ | Weight vector | Learned parameters controlling feature importance |
| $b$ | Bias term | Allows the decision boundary to shift away from the origin |
| $\mathbf{w}^T\mathbf{x} + b$ | Affine transformation (weighted sum + bias) | The pre-activation value — determines which side of the decision boundary the input lies on |
| $y$ | Binary output (0 or 1) | The final prediction after thresholding |

---

### Formal Definition: Multilayer Perceptron (MLP)

For a network with one hidden layer using an element-wise activation function $\sigma$:

$$
\begin{aligned}
\mathbf{h} &= \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \\
\hat{y} &= \sigma_{\text{out}}(\mathbf{W}_2 \mathbf{h} + \mathbf{b}_2)
\end{aligned}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{W}_1 \in \mathbb{R}^{d_h \times d_{\text{in}}}$ | Weight matrix from input to hidden layer | Each row corresponds to one hidden neuron's weights across all inputs |
| $\mathbf{b}_1 \in \mathbb{R}^{d_h}$ | Bias vector for hidden layer | Each hidden neuron has its own bias |
| $\mathbf{h} \in \mathbb{R}^{d_h}$ | Hidden layer activation vector | The learned intermediate representation |
| $\sigma$ | Non-linear activation function (e.g., ReLU, sigmoid) | Without it, the network collapses to a linear transformation |
| $\mathbf{W}_2 \in \mathbb{R}^{d_{\text{out}} \times d_h}$ | Weight matrix from hidden to output layer | Maps the representation to the final prediction |
| $\mathbf{b}_2 \in \mathbb{R}^{d_{\text{out}}}$ | Bias vector for output layer | Final per-output shift |
| $\sigma_{\text{out}}$ | Output activation (depends on task) | Sigmoid, softmax, or linear |
| $\hat{y}$ | Final prediction | The network's output |

> **ML Connection:** The weights and biases are the "knobs" of the network. Training adjusts these knobs so the network's squiggle fits the training data. Every parameter is learned via gradient-based optimization — there is no hand-crafting.

---

## 2. Forward Pass: Weighted Sum and Activation

### Motivation and Intuition

The forward pass is the process of computing a prediction from input to output. Data flows in one direction: input features are multiplied by weights, summed with a bias, run through an activation function, and the result becomes the input to the next layer. This is also called **forward propagation**.

### The Weighted Sum (Pre-Activation)

For a single neuron receiving $n$ inputs:

$$
z = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b = \sum_{i=1}^{n} w_i x_i + b
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $z$ | Pre-activation (logit) | The raw signal before the activation function |
| $x_i$ | Input feature $i$ | Data from the previous layer or raw input |
| $w_i$ | Weight for input $i$ | Controls how much input $i$ influences the neuron |
| $b$ | Bias term | Shifts the activation function left or right |
| $\sum_{i=1}^{n} w_i x_i$ | Dot product of weights and inputs | Aggregates all weighted signals into a single scalar |

### The Sigmoid Activation

The sigmoid function squashes any real number into the range $(0, 1)$:

$$
a = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\sigma(z)$ | Sigmoid output (activation) | Smooth S-curve between 0 and 1 — interpretable as a probability |
| $e^{-z}$ | Exponential decay | When $z$ is large positive, $e^{-z} \approx 0$ and $\sigma(z) \approx 1$; when $z$ is large negative, $\sigma(z) \approx 0$ |
| $a$ | Activation output | The neuron's firing rate, passed to the next layer |

### Worked Numerical Example

Suppose we have a single neuron with 2 inputs: $x_1 = 0.5$, $x_2 = -0.3$, weights $w_1 = 0.8$, $w_2 = -0.4$, and bias $b = 0.1$.

**Step 1 — Weighted sum:**

$$
z = 0.8 \cdot 0.5 + (-0.4) \cdot (-0.3) + 0.1 = 0.4 + 0.12 + 0.1 = 0.62
$$

**Step 2 — Sigmoid activation:**

$$
a = \sigma(0.62) = \frac{1}{1 + e^{-0.62}} = \frac{1}{1 + 0.538} = \frac{1}{1.538} \approx 0.650
$$

The neuron outputs $0.650$ — a moderate firing rate, closest to the steep region of the sigmoid curve.

---

### Forward Pass for a Mini-Network (1 Input, 2 Hidden, 1 Output)

**Step 1 — Hidden layer weighted sums:**

$$
z_1 = w_1 x + b_1, \quad z_2 = w_2 x + b_2
$$

**Step 2 — Hidden layer activations (sigmoid):**

$$
h_1 = \sigma(z_1), \quad h_2 = \sigma(z_2)
$$

**Step 3 — Output weighted sum:**

$$
z_3 = w_3 h_1 + w_4 h_2 + b_3
$$

**Step 4 — Output activation:**

$$
\hat{y} = \sigma(z_3)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $h_1, h_2$ | Hidden layer activations | Each neuron bends the input into a different curve shape |
| $w_1, w_2$ | Input-to-hidden weights | Control how the input is stretched/flipped before hitting each hidden neuron |
| $w_3, w_4$ | Hidden-to-output weights | Combine the two shaped curves into a single prediction |
| $b_1, b_2, b_3$ | Biases | Shift each activation independently for better data fitting |

> **ML Connection:** The forward pass is used at both training time (to compute predictions for the loss) and inference time (to make predictions on new data). During training, we cache all intermediate values ($z_1, h_1, z_2, h_2, z_3$) because backpropagation needs them.

---

### Python: Forward Pass from Scratch

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_pass(x, w1, b1, w2, b2, w3, b3, w4, b4):
    # Hidden layer
    z1 = w1 * x + b1
    h1 = sigmoid(z1)
    z2 = w2 * x + b2
    h2 = sigmoid(z2)
    # Output layer
    z3 = w3 * h1 + w4 * h2 + b3
    y_pred = sigmoid(z3)
    return y_pred, (z1, h1, z2, h2, z3)

x = 0.5
w1, b1 = 1.70, -0.85
w2, b2 = -1.70, 0.85
w3, b3, w4 = 1.00, 0.00, -1.00

y_pred, cache = forward_pass(x, w1, b1, w2, b2, w3, b3, w4, b4)
print(f"Prediction: {y_pred:.4f}")
```

---

## 3. Loss Functions: Measuring Prediction Error

### Motivation and Intuition

After the forward pass, we have a prediction $\hat{y}$. We need to quantify how wrong it is. The **loss function** (also called cost or error function) takes the prediction and the true value $y$ and returns a scalar that measures their discrepancy. Training minimizes this scalar.

### Mean Squared Error (MSE)

For regression problems, the most common loss is MSE. For a single data point:

$$
L = \frac{1}{2}(y - \hat{y})^2
$$

(The $\frac{1}{2}$ is a convenience factor that cancels the 2 from the derivative.)

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $L$ | Loss (scalar) | The error we want to minimize |
| $y$ | True target value | Ground truth from the training data |
| $\hat{y}$ | Predicted value | The network's output (depends on all parameters) |
| $(y - \hat{y})^2$ | Squared residual | Large errors are penalized quadratically |

For a batch of $n$ samples:

$$
L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

### Worked Example (MSE)

True value $y = 1.0$, predicted $\hat{y} = 0.65$:

$$
L = \frac{1}{2}(1.0 - 0.65)^2 = \frac{1}{2}(0.35)^2 = \frac{1}{2}(0.1225) = 0.06125
$$

**Derivative** (needed for backpropagation):

$$
\frac{\partial L}{\partial \hat{y}} = \hat{y} - y = 0.65 - 1.0 = -0.35
$$

> **ML Connection:** The choice of loss function depends on the task. MSE for regression, binary cross-entropy for binary classification, categorical cross-entropy for multi-class classification. The loss drives the entire learning process — it is the signal that backpropagation uses to adjust every parameter.

---

## 4. The Chain Rule: Essential Calculus for Backpropagation

### Motivation and Intuition

The output of a neural network is a composition of many functions. To compute how much a weight deep in the network contributed to the loss, we must differentiate through this chain of operations. The chain rule from calculus is the tool that makes this possible.

Think of it like a factory assembly line. If the final product is defective, you trace backward: the last station feeds into quality control, the second-to-last feeds into the last, and so on. The chain rule lets you measure how much each station contributed to the defect, given how each station affects the next.

### Single-Variable Chain Rule

If $y = f(u)$ and $u = g(x)$, then:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\frac{dy}{dx}$ | Derivative of $y$ w.r.t. $x$ | How the final output changes with the original input |
| $\frac{dy}{du}$ | Derivative of outer function | How the output changes with the intermediate value |
| $\frac{du}{dx}$ | Derivative of inner function | How the intermediate value changes with the input |

### Chain Rule Applied to a Neuron

For a single neuron: $y = \sigma(z)$ where $z = wx + b$. We want $\frac{\partial L}{\partial w}$:

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

Each local derivative is simple:

- $\frac{\partial L}{\partial a} = \hat{y} - y$ (derivative of MSE)
- $\frac{\partial a}{\partial z} = \sigma'(z) = \sigma(z)(1 - \sigma(z))$ (derivative of sigmoid)
- $\frac{\partial z}{\partial w} = x$ (derivative of $wx + b$)

### Multivariable Chain Rule

In neural networks, most functions have multiple inputs. For $z = w_1 x_1 + w_2 x_2 + b$ and $a = \sigma(z)$:

$$
\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w_1} = (\hat{y} - y) \cdot \sigma'(z) \cdot x_1
$$

### General Form for Deep Networks

For a loss $L$ depending on a weight $w$ deep in the network:

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h_k} \cdot \frac{\partial h_k}{\partial h_{k-1}} \cdot \dots \cdot \frac{\partial h_1}{\partial w}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\frac{\partial L}{\partial \hat{y}}$ | Gradient of loss w.r.t. prediction | The error signal starting at the output |
| $\frac{\partial \hat{y}}{\partial h_k}$ | Gradient through the output layer | How the prediction changes with the last hidden layer |
| $\frac{\partial h_{k}}{\partial h_{k-1}}$ | Gradient through hidden layer $k$ | How layer $k$ responds to layer $k-1$ |
| $\frac{\partial h_1}{\partial w}$ | Gradient of first hidden activation w.r.t. $w$ | The last link in the chain |

> **ML Connection:** The chain rule is the mathematical engine behind backpropagation. Every gradient in a neural network is a product of local derivatives. This modular structure means we can compute gradients for networks of arbitrary depth by multiplying simple terms.

---

## 5. Gradient Descent: The Optimization Algorithm

### Motivation and Intuition

After computing gradients via the chain rule, we need to update the parameters to reduce the loss. Gradient descent is the optimization algorithm that does this. Imagine standing on a hillside in thick fog. You cannot see the valley, but you can feel the slope. You take a step downhill, feel the slope again, and repeat until you reach the bottom.

### The Gradient Descent Update Rule

Each parameter $\theta$ (any weight $w$ or bias $b$) is updated as:

$$
\theta_{\text{new}} = \theta_{\text{old}} - \alpha \frac{\partial L}{\partial \theta}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\theta_{\text{old}}$ | Current parameter value | Where we are in parameter space |
| $\theta_{\text{new}}$ | Updated parameter | The new value after one step |
| $\alpha$ | Learning rate (hyperparameter) | Controls step size |
| $\frac{\partial L}{\partial \theta}$ | Gradient of loss w.r.t. $\theta$ | Direction and magnitude of steepest ascent (we move opposite) |
| $-\alpha \frac{\partial L}{\partial \theta}$ | The step | Negative gradient = move downhill |

### Gradient Descent Variants

| Algorithm | Data Per Update | Characteristics |
| :--- | :--- | :--- |
| Batch GD | All $n$ samples | Accurate but slow for large datasets |
| Stochastic GD (SGD) | 1 random sample | Fast updates but noisy convergence |
| Mini-Batch GD | Small random subset (e.g., 32) | Best of both worlds — the standard for deep learning |

### Worked Numerical Example

Suppose $w = 2.0$, $\frac{\partial L}{\partial w} = 0.5$, and $\alpha = 0.1$:

$$
w_{\text{new}} = 2.0 - 0.1 \cdot 0.5 = 2.0 - 0.05 = 1.95
$$

The gradient is positive ($0.5$), meaning increasing $w$ increases the loss. So we decrease $w$ by a small amount ($0.05$).

> **ML Connection:** Gradient descent is the optimizer used for virtually all neural network training. The gradients come from backpropagation; the update rule is gradient descent. Modern variants (Adam, RMSprop, SGD with momentum) improve upon basic gradient descent but share the same core idea.

---

## 6. Backpropagation: Full Derivation

### Motivation and Intuition

Backpropagation (short for "backward propagation of errors") computes the gradient of the loss with respect to every parameter in the network. It applies the chain rule systematically from the output layer backward to the input layer.

The key insight: intermediate values computed during the forward pass are reused during the backward pass. The error signal at each layer is computed once and then propagated backward through the weights.

### The Network (1 Input, 2 Hidden, 1 Output)

**Forward pass equations:**

$$
z_1 = w_1 x + b_1, \quad h_1 = \sigma(z_1)
$$

$$
z_2 = w_2 x + b_2, \quad h_2 = \sigma(z_2)
$$

$$
z_3 = w_3 h_1 + w_4 h_2 + b_3, \quad \hat{y} = \sigma(z_3)
$$

**Loss (MSE for a single point):**

$$
L = \frac{1}{2}(y - \hat{y})^2
$$

### Step 1: Output Error Signal

Compute the combined gradient of the loss with respect to $z_3$ (the output pre-activation). This is the **error signal** $\delta_3$ that propagates backward.

$$
\delta_3 = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_3} = (\hat{y} - y) \cdot \sigma(z_3)(1 - \sigma(z_3))
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\delta_3$ | Error signal at output node | Reusable for all parameters feeding into $z_3$ |
| $\hat{y} - y$ | Derivative of MSE | The raw prediction error |
| $\sigma(z_3)(1 - \sigma(z_3))$ | Sigmoid derivative at $z_3$ | Scales the error by the slope of the activation |

### Step 2: Output Layer Gradients

**Gradient for output bias $b_3$:**

$$
\frac{\partial L}{\partial b_3} = \delta_3 \cdot \frac{\partial z_3}{\partial b_3} = \delta_3 \cdot 1 = \delta_3
$$

**Gradient for output weight $w_3$:**

$$
\frac{\partial L}{\partial w_3} = \delta_3 \cdot \frac{\partial z_3}{\partial w_3} = \delta_3 \cdot h_1
$$

**Gradient for output weight $w_4$:**

$$
\frac{\partial L}{\partial w_4} = \delta_3 \cdot \frac{\partial z_3}{\partial w_4} = \delta_3 \cdot h_2
$$

| Parameter | Chain Rule Path | Gradient |
| :--- | :--- | :--- |
| $b_3$ | $L \to \hat{y} \to z_3 \to b_3$ | $\delta_3$ |
| $w_3$ | $L \to \hat{y} \to z_3 \to w_3$ | $\delta_3 \cdot h_1$ |
| $w_4$ | $L \to \hat{y} \to z_3 \to w_4$ | $\delta_3 \cdot h_2$ |

### Step 3: Hidden Layer Error Signals

To compute gradients for hidden layer parameters, we propagate the error signal backward through the weights $w_3$ and $w_4$, scaled by the hidden layer activation derivatives.

**Error signal for hidden node 1:**

$$
\delta_1 = \delta_3 \cdot w_3 \cdot \sigma(z_1)(1 - \sigma(z_1))
$$

**Error signal for hidden node 2:**

$$
\delta_2 = \delta_3 \cdot w_4 \cdot \sigma(z_2)(1 - \sigma(z_2))
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\delta_1$ | Error signal at hidden node 1 | Measures how much hidden node 1 contributed to the output error |
| $\delta_3 \cdot w_3$ | Error backpropagated through weight $w_3$ | The output error scaled by the connection strength |
| $\sigma(z_1)(1 - \sigma(z_1))$ | Sigmoid derivative at hidden node 1 | If saturated (near 0 or 1), the gradient vanishes |

### Step 4: Hidden Layer Gradients (Node 1)

**Gradient for $w_1$:**

$$
\frac{\partial L}{\partial w_1} = \delta_1 \cdot \frac{\partial z_1}{\partial w_1} = \delta_1 \cdot x
$$

$$
\frac{\partial L}{\partial w_1} = \delta_3 \cdot w_3 \cdot \sigma(z_1)(1 - \sigma(z_1)) \cdot x
$$

**Gradient for $b_1$:**

$$
\frac{\partial L}{\partial b_1} = \delta_1 \cdot \frac{\partial z_1}{\partial b_1} = \delta_1 \cdot 1 = \delta_1
$$

$$
\frac{\partial L}{\partial b_1} = \delta_3 \cdot w_3 \cdot \sigma(z_1)(1 - \sigma(z_1))
$$

### Step 4 (cont): Hidden Layer Gradients (Node 2)

**Gradient for $w_2$:**

$$
\frac{\partial L}{\partial w_2} = \delta_2 \cdot \frac{\partial z_2}{\partial w_2} = \delta_2 \cdot x = \delta_3 \cdot w_4 \cdot \sigma(z_2)(1 - \sigma(z_2)) \cdot x
$$

**Gradient for $b_2$:**

$$
\frac{\partial L}{\partial b_2} = \delta_2 = \delta_3 \cdot w_4 \cdot \sigma(z_2)(1 - \sigma(z_2))
$$

### Complete Gradient Table

| Parameter | Gradient Formula |
| :--- | :--- |
| $b_3$ | $\delta_3$ |
| $w_3$ | $\delta_3 \cdot h_1$ |
| $w_4$ | $\delta_3 \cdot h_2$ |
| $b_1$ | $\delta_3 \cdot w_3 \cdot \sigma(z_1)(1 - \sigma(z_1))$ |
| $w_1$ | $\delta_3 \cdot w_3 \cdot \sigma(z_1)(1 - \sigma(z_1)) \cdot x$ |
| $b_2$ | $\delta_3 \cdot w_4 \cdot \sigma(z_2)(1 - \sigma(z_2))$ |
| $w_2$ | $\delta_3 \cdot w_4 \cdot \sigma(z_2)(1 - \sigma(z_2)) \cdot x$ |

Where $\delta_3 = (\hat{y} - y) \cdot \sigma(z_3)(1 - \sigma(z_3))$.

### Step 5: Gradient Descent Updates

After computing all gradients, update every parameter:

$$
w_1 \leftarrow w_1 - \alpha \cdot \frac{\partial L}{\partial w_1}, \quad b_1 \leftarrow b_1 - \alpha \cdot \frac{\partial L}{\partial b_1}
$$

$$
w_2 \leftarrow w_2 - \alpha \cdot \frac{\partial L}{\partial w_2}, \quad b_2 \leftarrow b_2 - \alpha \cdot \frac{\partial L}{\partial b_2}
$$

$$
w_3 \leftarrow w_3 - \alpha \cdot \frac{\partial L}{\partial w_3}, \quad w_4 \leftarrow w_4 - \alpha \cdot \frac{\partial L}{\partial w_4}, \quad b_3 \leftarrow b_3 - \alpha \cdot \frac{\partial L}{\partial b_3}
$$

### Worked Numerical Example

Let us trace one complete training iteration with concrete numbers.

**Initial state:**
- $x = 0.5$, $y = 1.0$
- $w_1 = 1.70, b_1 = -0.85, w_2 = -1.70, b_2 = 0.85, w_3 = 1.00, w_4 = -1.00, b_3 = 0.00$
- $\alpha = 0.1$

**Forward pass:**

$$
z_1 = 1.70 \cdot 0.5 + (-0.85) = 0.85 - 0.85 = 0.0, \quad h_1 = \sigma(0) = 0.5
$$

$$
z_2 = -1.70 \cdot 0.5 + 0.85 = -0.85 + 0.85 = 0.0, \quad h_2 = \sigma(0) = 0.5
$$

$$
z_3 = 1.00 \cdot 0.5 + (-1.00) \cdot 0.5 + 0.00 = 0.0, \quad \hat{y} = \sigma(0) = 0.5
$$

**Loss:** $L = \frac{1}{2}(1.0 - 0.5)^2 = 0.125$

**Backward pass:**

$$
\delta_3 = (0.5 - 1.0) \cdot 0.5(1 - 0.5) = (-0.5) \cdot 0.25 = -0.125
$$

$$
\frac{\partial L}{\partial b_3} = -0.125, \quad \frac{\partial L}{\partial w_3} = -0.125 \cdot 0.5 = -0.0625, \quad \frac{\partial L}{\partial w_4} = -0.125 \cdot 0.5 = -0.0625
$$

$$
\delta_1 = -0.125 \cdot 1.00 \cdot 0.5(1 - 0.5) = -0.125 \cdot 0.25 = -0.03125
$$

$$
\frac{\partial L}{\partial w_1} = -0.03125 \cdot 0.5 = -0.015625, \quad \frac{\partial L}{\partial b_1} = -0.03125
$$

$$
\delta_2 = -0.125 \cdot (-1.00) \cdot 0.5(1 - 0.5) = 0.125 \cdot 0.25 = 0.03125
$$

$$
\frac{\partial L}{\partial w_2} = 0.03125 \cdot 0.5 = 0.015625, \quad \frac{\partial L}{\partial b_2} = 0.03125
$$

**Updates ($\alpha = 0.1$):**

$$
w_1 \leftarrow 1.70 - 0.1 \cdot (-0.015625) = 1.7015625
$$

$$
b_1 \leftarrow -0.85 - 0.1 \cdot (-0.03125) = -0.846875
$$

And similarly for all other parameters. After this update, the loss will be slightly lower. Repeating this process thousands of times drives the loss to a minimum.

### The Modular Pattern

The key insight that makes backpropagation scale to deep networks:

1. Compute the error signal $\delta_{\text{out}}$ at the output.
2. For each layer working backward: given the error signal $\delta_{\text{above}}$ from the layer above, compute:
   - Gradients for this layer's weights: $\delta_{\text{above}} \times \text{(input to this weight)}$
   - Gradients for this layer's biases: $\delta_{\text{above}}$
   - Error signal for the layer below: $\delta_{\text{above}} \times \text{(weight)} \times \text{(activation derivative)}$
3. Repeat until all gradients are computed.

---

### Python: Backpropagation from Scratch

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def train_one_step(x, y, params, alpha=0.1):
    w1, b1, w2, b2, w3, b3, w4 = params

    # Forward pass
    z1 = w1 * x + b1
    h1 = sigmoid(z1)
    z2 = w2 * x + b2
    h2 = sigmoid(z2)
    z3 = w3 * h1 + w4 * h2 + b3
    y_pred = sigmoid(z3)

    # Loss
    loss = 0.5 * (y - y_pred) ** 2

    # Backward pass
    delta_3 = (y_pred - y) * sigmoid_derivative(y_pred)
    dL_db3 = delta_3
    dL_dw3 = delta_3 * h1
    dL_dw4 = delta_3 * h2

    delta_1 = delta_3 * w3 * sigmoid_derivative(h1)
    dL_dw1 = delta_1 * x
    dL_db1 = delta_1

    delta_2 = delta_3 * w4 * sigmoid_derivative(h2)
    dL_dw2 = delta_2 * x
    dL_db2 = delta_2

    # Update
    w1 -= alpha * dL_dw1
    b1 -= alpha * dL_db1
    w2 -= alpha * dL_dw2
    b2 -= alpha * dL_db2
    w3 -= alpha * dL_dw3
    w4 -= alpha * dL_dw4
    b3 -= alpha * dL_db3

    new_params = (w1, b1, w2, b2, w3, b3, w4)
    return new_params, loss, y_pred

params = (1.70, -0.85, -1.70, 0.85, 1.00, 0.00, -1.00)
x, y = 0.5, 1.0

for epoch in range(5):
    params, loss, pred = train_one_step(x, y, params, alpha=0.1)
    print(f"Epoch {epoch+1}: Loss = {loss:.6f}, Prediction = {pred:.4f}")
```

---

## 7. Learning Rate, Convergence, and Local Minima

### Motivation and Intuition

Gradient descent is not magic — it can fail in several ways. The learning rate must be chosen carefully. The loss surface for neural networks is non-convex (bumpy), so gradient descent may get stuck in a local minimum or saddle point. Understanding these failure modes is essential for successful training.

### The Learning Rate

The learning rate $\alpha$ is the most important hyperparameter:

| Learning Rate | Behavior | Outcome |
| :--- | :--- | :--- |
| Too small | Tiny steps | Very slow convergence; may get stuck on plateaus |
| Just right | Steady downhill steps | Efficient convergence to a good minimum |
| Too large | Wild overshooting | Loss diverges (explodes to infinity) |

**Practical tips:**
- Start with $\alpha = 0.01$ or $0.001$ and monitor the loss curve.
- If loss oscillates or increases, reduce $\alpha$.
- If loss decreases very slowly, increase $\alpha$.
- Use **learning rate schedules** (e.g., step decay, exponential decay, cosine annealing) that reduce $\alpha$ over time.

### Convergence

Convergence means the loss has stopped decreasing significantly. Signs of convergence:

- The loss curve flattens (reaches a plateau).
- Validation loss stops improving (may even start increasing — sign of overfitting).
- Gradients become very small (near zero).

**Stopping criteria:**
- Fixed number of epochs.
- Early stopping: stop when validation loss has not improved for $n$ consecutive epochs (patience).

### Local Minima and Saddle Points

| Problem | Description | How to Handle |
| :--- | :--- | :--- |
| **Local Minimum** | A valley that is not the lowest point in the landscape | Use momentum (helps roll out of shallow minima); try random restarts |
| **Saddle Point** | A flat region where gradient is zero in some directions but not others | Common in high dimensions; SGD noise usually escapes |
| **Plateau** | A flat region with very small gradients | Increase learning rate temporarily; use adaptive methods (Adam) |

> **ML Connection:** In practice, for large neural networks, local minima are not a serious problem. Most local minima are close to the global minimum in terms of loss value. Saddle points are more common but SGD's stochasticity (random mini-batches) provides enough noise to escape them.

---

### Python: Training Loop with Convergence Tracking

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def train_network(x, y, params, alpha=0.1, epochs=100):
    loss_history = []
    for epoch in range(epochs):
        params, loss, pred = train_one_step(x, y, params, alpha)
        loss_history.append(loss)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
    return params, loss_history

# Convert to batch training later
x_train = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
y_train = np.array([0.1, 0.3, 0.7, 0.9, 1.0])

def train_batch(x_data, y_data, params, alpha=0.1, epochs=200):
    for epoch in range(epochs):
        total_loss = 0.0
        for xi, yi in zip(x_data, y_data):
            params, loss, _ = train_one_step(xi, yi, params, alpha)
            total_loss += loss
        if epoch % 40 == 0:
            print(f"Epoch {epoch}: Avg Loss = {total_loss/len(x_data):.6f}")
    return params

params = (1.70, -0.85, -1.70, 0.85, 1.00, 0.00, -1.00)
params = train_batch(x_train, y_train, params, alpha=0.1, epochs=200)
print("Training complete.")
```

> **Check your intuition:** If the gradient $\frac{\partial L}{\partial w} = -0.2$ and $\alpha = 0.5$, what is the new weight if the old weight was $w = 3.0$? In which direction does $w$ move?

<details>
<summary>Answer</summary>
$w_{\text{new}} = 3.0 - 0.5 \cdot (-0.2) = 3.0 + 0.1 = 3.1$. Since the gradient is negative (downhill to the right), we move right (increase $w$).
</details>

---

## Prerequisites and Further Reading

- **Previous:** Linear Regression, Logistic Regression (from Supervised Learning)
- **Next:** 02-Activation-Functions (impact of different activations on training)
- **Related:** L03 The Chain Rule (calculus foundation), L04 Gradient Descent (optimization), L05-L07 Backpropagation (full derivation)
- **Foundational:** Matrix multiplication, partial derivatives, chain rule from 00-Mathematics-Foundation
- **Further:** Deep Learning (Goodfellow et al., 2016) Chapters 6-8
