## L02 Neural Networks Part 1 - Essential Main Ideas

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. Neural Network Architecture

### Motivation and Intuition

A neural network takes inputs, transforms them through a series of layers, and produces an output. The simplest architecture has three types of layers:

- **Input Layer** — Each node represents one feature (e.g., drug dosage).
- **Hidden Layer** — One or more layers between input and output. Each node applies an activation function to a weighted sum of the inputs.
- **Output Layer** — Produces the final prediction.

Each connection between nodes has a **weight** that controls how much influence one node has on the next. Each node (except input nodes) also has a **bias** that shifts the activation function, giving the network flexibility to fit data that does not pass through the origin.

### Anatomy of a Single Neuron

A single neuron in the hidden layer does two things:

1. **Compute the weighted sum** of its inputs plus the bias.
2. **Apply an activation function** to that sum to produce a non-linear output.

---

## 2. Weighted Sum and the Bias

### The Weighted Sum

For a neuron receiving $n$ inputs, the weighted sum (also called the logit or affine transformation) is:

$$
z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b = \sum_{i=1}^{n} w_i x_i + b
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $z$ | Weighted sum plus bias (pre-activation) | The raw signal before the activation function — determines the x-axis coordinate for the activation curve |
| $x_i$ | Input feature $i$ | The data coming into the neuron from the previous layer or the raw input |
| $w_i$ | Weight for input $i$ | Controls how much input $i$ influences the neuron — learned during training |
| $b$ | Bias term | Shifts the activation function left or right — allows the network to fit data not centered at the origin |
| $\sum_{i=1}^{n} w_i x_i$ | Dot product of weights and inputs | Aggregates all weighted input signals into a single value |

### The Bias as an Intercept

Think of $b$ like the intercept in linear regression $y = mx + b$. Without the bias, every activation function would be forced through the origin, severely limiting what shapes the network can learn.

---

## 3. Activation Functions — The Sigmoid

### Why We Need Non-Linearity

Without an activation function, stacking layers is equivalent to a single linear transformation — no more powerful than linear regression. Activation functions introduce the bends and curves that let neural networks fit non-linear data.

### The Sigmoid Activation Function

The sigmoid function squashes any real number into the range $(0, 1)$:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\sigma(z)$ | Sigmoid output (activation) | A smooth S-shaped curve between 0 and 1 — interprets the weighted sum as a "firing rate" |
| $z$ | Weighted sum + bias (pre-activation) | The input to the sigmoid — determines where on the S-curve we land |
| $e^{-z}$ | Exponential decay term | When $z$ is large positive, $e^{-z} \approx 0$ and $\sigma(z) \approx 1$. When $z$ is large negative, $e^{-z}$ is huge and $\sigma(z) \approx 0$. When $z=0$, $\sigma(0)=0.5$ |

### Why the Sigmoid Shape Matters

The S-shape creates a smooth transition between "off" (0) and "on" (1). The steep region in the middle means small changes in $z$ around zero produce large changes in the output — perfect for decision boundaries. The flat tails mean extreme inputs get capped, providing natural saturation.

---

## 4. Forward Propagation — Making a Prediction

### One Step at a Time

Forward propagation is the process of moving data from the input layer through the hidden layer to the output layer. For a single hidden layer with two nodes:

**Step 1:** Compute the weighted sum for each hidden node:

$$
z_1 = w_1 x + b_1 \quad,\quad z_2 = w_2 x + b_2
$$

**Step 2:** Apply the sigmoid to each:

$$
h_1 = \sigma(z_1) \quad,\quad h_2 = \sigma(z_2)
$$

**Step 3:** Combine hidden outputs with output weights and bias:

$$
z_3 = w_3 h_1 + w_4 h_2 + b_3
$$

**Step 4:** Apply output activation (e.g., sigmoid for binary classification):

$$
\hat{y} = \sigma(z_3)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $h_1, h_2$ | Hidden layer activations | The transformed representations — each node bends the original input into a different shape |
| $w_1, w_2$ | Weights from input to hidden nodes | Control how the input stretches/flips each activation function |
| $w_3, w_4$ | Weights from hidden to output | Combine the two shaped curves into a single prediction |
| $b_1, b_2$ | Hidden biases | Shift each activation function left/right independently |
| $b_3$ | Output bias | Final vertical shift of the entire squiggle |

---

## 5. Putting It All Together — How the Network Fits a Squiggle

Each hidden node with a sigmoid activation produces an S-shaped curve. The weights stretch or flip that curve. The biases shift it left or right. When you add two or more of these transformed S-curves together, you get a flexible "squiggle" that can fit all sorts of patterns:

- One node captures the upward slope.
- Another node captures the downward slope.
- The output bias shifts everything up or down.

The exact combination of weights and biases determines the final shape. Training adjusts these numbers until the squiggle matches the data.

---

## 6. Python Code — A Tiny Neural Network from Scratch

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagation(x, w1, b1, w2, b2, w3, b3, w4, b4):
    # Hidden layer — two nodes
    z1 = w1 * x + b1
    h1 = sigmoid(z1)

    z2 = w2 * x + b2
    h2 = sigmoid(z2)

    # Output layer
    z3 = w3 * h1 + w4 * h2 + b3
    y_pred = sigmoid(z3)

    return y_pred, h1, h2

# Example: predict effectiveness for a medium dosage
dosage = 0.5  # 0 = low, 1 = high

# Randomly initialized weights and biases
w1, b1 = 1.70, -0.85
w2, b2 = -1.70, 0.85
w3, b3, w4 = 1.00, 0.00, -1.00

prediction, h1_out, h2_out = forward_propagation(dosage, w1, b1, w2, b2, w3, b3, w4, b4)
print(f"Predicted effectiveness: {prediction:.4f}")
print(f"Hidden node 1: {h1_out:.4f}, Hidden node 2: {h2_out:.4f}")
```

This tiny network has two hidden nodes, four weights, and three biases — just 7 parameters total. Yet with the right values, it can fit a bell-shaped "effective only at medium dosage" curve. That is the power of stacking non-linear transformations.

---

> **Check your intuition:** What would happen to the network's output if we set all weights to zero? What if we set all biases to zero?

---

## Prerequisites and Further Reading

- **Previous:** L01 Neural Networks Part 0 — Not Scary (conceptual overview)
- **Next:** L03 The Chain Rule (the math behind backpropagation)
- **Related:** Logistic Regression (sigmoid function origins)
