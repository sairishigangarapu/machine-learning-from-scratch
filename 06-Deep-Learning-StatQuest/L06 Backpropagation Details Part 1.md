## L06 Backpropagation Details Part 1

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. Setting Up the Problem

### Motivation and Intuition

In L05, we optimized just the output bias $b_3$ while assuming all other parameters were already optimal. Now, suppose we do not know the optimal values for the last two weights $w_3$ and $w_4$ either. We will compute gradients for $w_3$, $w_4$, and $b_3$ simultaneously, using the chain rule, and then update them all with gradient descent.

This is the natural next step: instead of optimizing one parameter at a time, we optimize a whole layer at a time. The pattern extends to any number of parameters.

### The Network

Same network as before — one input $x$, two hidden sigmoid nodes, one output sigmoid node:

$$
z_1 = w_1 x + b_1, \quad h_1 = \sigma(z_1)
$$

$$
z_2 = w_2 x + b_2, \quad h_2 = \sigma(z_2)
$$

$$
z_3 = w_3 h_1 + w_4 h_2 + b_3, \quad \hat{y} = \sigma(z_3)
$$

We assume $w_1, b_1, w_2, b_2$ are already optimal. We initialize $w_3, w_4, b_3$ with random values.

---

## 2. The Loss and the First Gradient Link

### Mean Squared Error

For a single data point:

$$
L = (y - \hat{y})^2
$$

### Common Starting Point

Every gradient for every parameter in the output layer shares the first part of the chain: $\frac{\partial L}{\partial \hat{y}}$ and $\frac{\partial \hat{y}}{\partial z_3}$.

Let us compute these once and reuse them:

$$
\frac{\partial L}{\partial \hat{y}} = -2(y - \hat{y})
$$

$$
\frac{\partial \hat{y}}{\partial z_3} = \sigma(z_3)(1 - \sigma(z_3))
$$

Define the **error signal** (sometimes called "delta") at the output:

$$
\delta_3 = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_3} = -2(y - \hat{y}) \cdot \sigma(z_3)(1 - \sigma(z_3))
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\delta_3$ | Error signal at the output node | The combined gradient of the loss w.r.t. the pre-activation $z_3$ — reusable for all parameters that feed into $z_3$ |
| $-2(y - \hat{y})$ | Loss derivative | How much the MSE changes when the prediction changes |
| $\sigma(z_3)(1 - \sigma(z_3))$ | Sigmoid derivative at $z_3$ | The slope of the S-curve at the current pre-activation value |

---

## 3. Gradient for $b_3$ (Output Bias)

The chain: $L \to \hat{y} \to z_3 \to b_3$

$$
\frac{\partial L}{\partial b_3} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_3} \cdot \frac{\partial z_3}{\partial b_3}
$$

Since $\frac{\partial z_3}{\partial b_3} = 1$:

$$
\frac{\partial L}{\partial b_3} = \delta_3
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\frac{\partial z_3}{\partial b_3}$ | Derivative of $z_3$ w.r.t. $b_3$ | Always 1 because $z_3 = w_3 h_1 + w_4 h_2 + b_3$ — the bias adds directly |
| $\frac{\partial L}{\partial b_3}$ | Gradient for output bias | The error signal itself — $b_3$ gets the full gradient with no scaling |

---

## 4. Gradients for $w_3$ and $w_4$ (Output Weights)

### Gradient for $w_3$

The chain: $L \to \hat{y} \to z_3 \to w_3$

$$
\frac{\partial L}{\partial w_3} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_3} \cdot \frac{\partial z_3}{\partial w_3}
$$

Since $\frac{\partial z_3}{\partial w_3} = h_1$:

$$
\frac{\partial L}{\partial w_3} = \delta_3 \cdot h_1
$$

### Gradient for $w_4$

Similarly:

$$
\frac{\partial L}{\partial w_4} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_3} \cdot \frac{\partial z_3}{\partial w_4} = \delta_3 \cdot h_2
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\frac{\partial z_3}{\partial w_3}$ | Derivative of $z_3$ w.r.t. $w_3$ | Equals $h_1$ — because $z_3 = w_3 h_1 + \dots$ |
| $\frac{\partial L}{\partial w_3}$ | Gradient for weight $w_3$ | Error signal times the hidden activation that feeds into this weight — larger activations get larger weight updates |
| $h_1$ | Activation of hidden node 1 | The input to this weight from the hidden layer — scales the gradient |

### The Pattern for Output Weights

For any weight connecting a node to the output:

$$
\frac{\partial L}{\partial w_{\text{out}}} = \delta_{\text{out}} \cdot (\text{input to that weight from previous layer})
$$

---

## 5. Gradient Descent Updates

Once we have all three gradients, we update:

$$
w_3 \leftarrow w_3 - \alpha \cdot (\delta_3 \cdot h_1)
$$

$$
w_4 \leftarrow w_4 - \alpha \cdot (\delta_3 \cdot h_2)
$$

$$
b_3 \leftarrow b_3 - \alpha \cdot \delta_3
$$

Each parameter moves in the direction that reduces the loss. The size of the move depends on both the error signal and the local input (for weights) or just the error signal (for biases).

---

## 6. Summary of the Backward Pass So Far

| Parameter | Chain Rule Path | Gradient Formula |
| :--- | :--- | :--- |
| $b_3$ | $L \to \hat{y} \to z_3 \to b_3$ | $\delta_3$ |
| $w_3$ | $L \to \hat{y} \to z_3 \to w_3$ | $\delta_3 \cdot h_1$ |
| $w_4$ | $L \to \hat{y} \to z_3 \to w_4$ | $\delta_3 \cdot h_2$ |

These three gradients form a complete update step for the output layer. In L07, we will continue backward through the hidden layer to compute gradients for $w_1, b_1, w_2, b_2$ — the full backpropagation.

---

> **Check your intuition:** If $h_1 = 0$, what happens to the gradient for $w_3$? Why does this make intuitive sense?

---

## Prerequisites and Further Reading

- **Previous:** L05 Backpropagation Main Ideas
- **Next:** L07 Backpropagation Details Part 2 — gradients through hidden layers
- **Related:** L04 Gradient Descent (how we use the gradients we computed here)
