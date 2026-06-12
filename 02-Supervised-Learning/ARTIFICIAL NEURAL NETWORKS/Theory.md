# Artificial Neural Networks: From Neurons to Deep Learning

## 1. Biological Inspiration

The **Artificial Neural Network (ANN)** is inspired by the biological neuron in the human brain:

| Biological Neuron | Artificial Neuron |
| :--- | :--- |
| **Dendrites** — receive signals from other cells | **Inputs** ($x_1, x_2, \dots, x_n$) |
| **Soma** — processes the signal | **Weighted Sum + Activation** |
| **Axon** — transmits the output | **Output** ($y$) |
| **Synapse** — connection strength between neurons | **Weight** ($w_i$) |

> The human brain has ~100 billion neurons with ~100 trillion connections. ANNs are a *massively simplified* approximation.

---

## 2. The MP Neuron (McCulloch-Pitts, 1943)

The first mathematical model of a neuron. It uses **thresholding logic**:

$$
y = \begin{cases} 1 & \text{if } \sum_{i=1}^{n} x_i \geq \theta \\ 0 & \text{otherwise} \end{cases}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $x_i$ | The $i$-th binary input feature (0 or 1) | Each input represents a binary signal from a predecessor neuron |
| $n$ | Total number of input features | Determines the dimensionality of the input space |
| $\theta$ | Threshold parameter | If the sum of inputs meets or exceeds this value, the neuron fires (outputs 1) |
| $y$ | Binary output of the neuron | 1 = neuron fires, 0 = neuron stays silent |
| $\sum_{i=1}^{n} x_i$ | Sum of all inputs | Aggregates all incoming signals into a single value for comparison against the threshold |

### Limitations
* Inputs must be binary (0 or 1).
* All inputs are weighted equally (no individual $w_i$).
* Threshold $\theta$ is hand-coded, not learned.
* Can only represent **linearly separable** functions (e.g., AND, OR but **not XOR**).

---

## 3. The Perceptron (Rosenblatt, 1958)

A generalization of the MP neuron that introduces **learnable weights** and a **bias**:

$$
y = \begin{cases} 1 & \text{if } \mathbf{w}^T\mathbf{x} + b \geq 0 \\ 0 & \text{otherwise} \end{cases}
$$

| Component | Role |
| :--- | :--- |
| $w_i$ | Weight for feature $x_i$ — learned from data |
| $b$ | Bias (equivalent to $-\theta$) — allows the decision boundary to shift |
| $\mathbf{w}^T\mathbf{x} + b$ | **Decision boundary** — a hyperplane in feature space |

### Perceptron Learning Algorithm

```
Initialize w = 0, b = 0
Repeat until convergence:
 For each training example (x_i, y_i):
 ŷ = predict(w, b, x_i)
 if ŷ != y_i:
 w = w + learning_rate * (y_i - ŷ) * x_i
 b = b + learning_rate * (y_i - ŷ)
```

> **Convergence Guarantee:** If the data is linearly separable, the Perceptron algorithm is guaranteed to converge. If not (e.g., XOR), it will loop forever.

---

## 4. The XOR Problem: Why We Need Multilayer Networks

The XOR function (output 1 iff exactly one input is 1) is **not linearly separable** — no single line can separate the classes:

| $x_1$ | $x_2$ | XOR |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

A single perceptron **cannot learn XOR**. This limitation motivated the **Multilayer Perceptron (MLP)** — stacking layers of perceptrons with non-linear activations.

---

## 5. Multilayer Perceptron (MLP) Architecture

```
Input Layer Hidden Layer(s) Output Layer
 x₁ ────────→ h₁ ────────→ ŷ
 x₂ ────────→ h₂ ────────→
 x₃ ────────→ h₃ ────────→
```

### Key Definitions
* **Feedforward:** Information flows one direction (input → output). No cycles.
* **Fully Connected:** Every neuron in layer $l$ connects to every neuron in layer $l+1$.
* **Hidden Layers:** Layers between input and output. A network with ≥2 hidden layers is a **Deep Neural Network**.

### Forward Pass (Single Hidden Layer)

$$
\begin{aligned}
\mathbf{h} &= \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \quad \text{(hidden layer)} \\
\hat{y} &= \sigma(\mathbf{W}_2 \mathbf{h} + \mathbf{b}_2) \quad \text{(output layer)}
\end{aligned}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{x}$ | Input feature vector | The raw data fed into the network |
| $\mathbf{W}_1$ | Weight matrix from input to hidden layer | Learns which input features are important and how to combine them |
| $\mathbf{b}_1$ | Bias vector for the hidden layer | Allows the activation function to shift left/right, fitting data that doesn't pass through the origin |
| $\mathbf{h}$ | Hidden layer output vector | The learned intermediate representation — each element captures a different pattern in the input |
| $\mathbf{W}_2$ | Weight matrix from hidden to output layer | Maps the learned representation to the final prediction |
| $\mathbf{b}_2$ | Bias vector for the output layer | Final adjustment to the output before activation |
| $\hat{y}$ | Predicted output | The network's final prediction after all transformations |
| $\sigma$ | Non-linear activation function (e.g., ReLU, sigmoid) | Without it, the entire network would be a single linear transformation — no more powerful than logistic regression |

---

## 6. Activation Functions

Without non-linearity, stacking layers is equivalent to a single linear transformation. Activation functions introduce the non-linearity that allows MLPs to learn complex patterns.

| Function | Formula | Range | Used When |
| :--- | :--- | :--- | :--- |
| **Sigmoid** | $\sigma(z) = \frac{1}{1+e^{-z}}$ | (0, 1) | Binary output layer |
| **Tanh** | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | (-1, 1) | Hidden layers (centered output) |
| **ReLU** | $f(z) = \max(0, z)$ | [0, ∞) | **Default for hidden layers** (2026 standard) |
| **Leaky ReLU** | $f(z) = \max(\alpha z, z)$ | (-∞, ∞) | When ReLU "dying neuron" is a problem |
| **Softmax** | $\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | (0, 1), sums to 1 | Multi-class output layer |

### Why ReLU Dominates
* Computationally cheap (just a threshold).
* Avoids vanishing gradient problem (for $z > 0$, gradient = 1).
* Induces sparsity (many neurons output exactly 0).

---

## 7. Backpropagation: How ANNs Learn

**Backpropagation** (Rumelhart, Hinton, Williams, 1986) computes how much each weight contributed to the error, using the **chain rule** of calculus.

### The Algorithm

**Step 1 — Forward Pass:** Compute predictions $\hat{y}$ and loss $L$.

**Step 2 — Backward Pass:** Compute gradients $\frac{\partial L}{\partial w}$ for every weight using the chain rule:

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h} \cdot \frac{\partial h}{\partial w}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $L$ | Loss function (e.g., cross-entropy, MSE) | Quantifies how wrong the prediction is — the value we want to minimize |
| $\frac{\partial L}{\partial \hat{y}}$ | Gradient of loss w.r.t. prediction | How much the loss changes when the prediction changes — the "error signal" |
| $\frac{\partial \hat{y}}{\partial h}$ | Gradient of prediction w.r.t. hidden unit | How the prediction changes when hidden activations change |
| $\frac{\partial h}{\partial w}$ | Gradient of hidden unit w.r.t. weight | How the hidden activation changes when a weight changes |
| $w$ | A single model weight | One of the thousands/millions of parameters being optimized |

**Step 3 — Update:** Move each weight in the direction that reduces the loss:

$$
w \leftarrow w - \alpha \cdot \frac{\partial L}{\partial w}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\alpha$ | Learning rate (hyperparameter) | Controls step size — too large causes overshooting/divergence, too small causes slow convergence |
| $w$ | Current weight value | The parameter being updated |
| $\frac{\partial L}{\partial w}$ | Gradient of loss w.r.t. this weight | The direction and magnitude of the steepest increase in loss |
| $w - \alpha \cdot \frac{\partial L}{\partial w}$ | Updated weight | Moving against the gradient reduces the loss (gradient descent) |

### Intuition
Backpropagation answers: *"If I nudge this weight slightly, how does the loss change?"* It propagates the error signal backward from output to input, layer by layer.

---

## 8. Loss Functions

| Task | Loss Function | Formula |
| :--- | :--- | :--- |
| Binary Classification | **Binary Cross-Entropy** | $-\left[y\log(\hat{y}) + (1-y)\log(1-\hat{y})\right]$ |
| Multi-class Classification | **Categorical Cross-Entropy** | $-\sum_{c=1}^{C} y_c \log(\hat{y}_c)$ |
| Regression | **Mean Squared Error** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ |

---

## 9. Code Example: XOR with MLP

```python
import numpy as np
from sklearn.neural_network import MLPClassifier

# XOR dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

# Single perceptron CANNOT learn XOR
from sklearn.linear_model import Perceptron
p = Perceptron()
p.fit(X, y)
print(f"Perceptron accuracy: {p.score(X, y):.2f}") # 0.50 (fails)

# MLP CAN learn XOR
mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000, random_state=42)
mlp.fit(X, y)
print(f"MLP accuracy: {mlp.score(X, y):.2f}") # 1.00 (succeeds)
print(f"Predictions: {mlp.predict(X)}")
```

---

## 10. Key Hyperparameters

| Hyperparameter | Effect |
| :--- | :--- |
| `hidden_layer_sizes` | Number and size of hidden layers. More layers = more capacity but risk overfitting. |
| `learning_rate` | Step size for weight updates. Too high → diverge; too low → slow. |
| `activation` | Non-linearity function. Default: `relu` (recommended). |
| `max_iter` | Training epochs. Monitor loss curve for convergence. |
| `batch_size` | Samples per gradient update. Smaller = noisier but faster. |

---

## 11. Advantages & Disadvantages

### Pros
* Can learn **non-linear** decision boundaries (unlike logistic regression).
* Universal approximators — can approximate any continuous function with enough neurons.
* Scale well with data and compute.

### Cons
* **Black box** — hard to interpret individual predictions.
* Require careful hyperparameter tuning.
* Prone to **overfitting** on small datasets.
* computationally expensive for very deep architectures.

---

**Previous:** [Random Forest](../RANDOM%20FOREST/Theory.md) | **Next:** [SVM](../SUPPORT%20VECTOR%20MACHINE%20(SVM)/Theory.md) | **Related:** [Bias-Variance Tradeoff](../../01-Core-Concepts/Bias-Variance-Tradeoff.md)
