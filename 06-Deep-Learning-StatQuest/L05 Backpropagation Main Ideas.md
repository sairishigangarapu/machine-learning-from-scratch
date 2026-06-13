## L05 Backpropagation Main Ideas

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. What Backpropagation Is

### Motivation and Intuition

Backpropagation (short for "backward propagation of errors") is the algorithm that trains neural networks. It answers the question: "How much did each weight and bias contribute to the final error?"

The idea is simple. Run the data forward through the network to get a prediction (forward pass). Compare the prediction to the true value and compute the loss. Then work backward from the loss, using the chain rule to compute how much each parameter contributed. Finally, use gradient descent to update each parameter in the direction that reduces the loss.

Think of it like a game of telephone played in reverse. The loss says "I need to change." The last layer heard it first, so it adjusts. Then the previous layer hears "the last layer changed, so you need to change too," and so on, all the way back to the first layer.

### The Two Passes

| Pass | Direction | What Happens |
| :--- | :--- | :--- |
| **Forward Pass** | Input → Output | Compute predictions and the loss |
| **Backward Pass** | Output → Input | Compute gradients of the loss with respect to every parameter using the chain rule |

---

## 2. The Forward Pass

### Computing Predictions

For a simple network with one input, two hidden nodes (using sigmoid), and one output:

$$
z_1 = w_1 x + b_1 \quad,\quad h_1 = \sigma(z_1)
$$

$$
z_2 = w_2 x + b_2 \quad,\quad h_2 = \sigma(z_2)
$$

$$
z_3 = w_3 h_1 + w_4 h_2 + b_3 \quad,\quad \hat{y} = \sigma(z_3)
$$

### The Loss

Using Mean Squared Error for a single data point:

$$
L = (y - \hat{y})^2
$$

The forward pass gives us all the intermediate values ($z_1, h_1, z_2, h_2, z_3, \hat{y}$) that we will need on the backward pass.

---

## 3. The Backward Pass — Chain Rule in Action

### Computing $\frac{\partial L}{\partial b_3}$ (Output Bias)

We start at the loss and work backward one step at a time.

**Step 1:** How does the loss change with the prediction $\hat{y}$?

$$
\frac{\partial L}{\partial \hat{y}} = -2(y - \hat{y})
$$

**Step 2:** How does $\hat{y}$ change with $z_3$ (the pre-activation)?

$$
\frac{\partial \hat{y}}{\partial z_3} = \sigma(z_3)(1 - \sigma(z_3))
$$

**Step 3:** How does $z_3$ change with $b_3$?

$$
\frac{\partial z_3}{\partial b_3} = 1
$$

**Step 4:** Multiply them together (chain rule):

$$
\frac{\partial L}{\partial b_3} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_3} \cdot \frac{\partial z_3}{\partial b_3} = -2(y - \hat{y}) \cdot \sigma(z_3)(1 - \sigma(z_3)) \cdot 1
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\frac{\partial L}{\partial \hat{y}}$ | Error signal from the loss | How much the loss changes when the prediction changes — our starting point |
| $\frac{\partial \hat{y}}{\partial z_3}$ | Derivative of output activation | How much the prediction changes when the pre-activation changes — the sigmoid slope at $z_3$ |
| $\frac{\partial z_3}{\partial b_3}$ | Derivative of weighted sum w.r.t. bias | Always 1 — the bias directly adds to $z_3$ |
| $\frac{\partial L}{\partial b_3}$ | Final gradient for $b_3$ | The product — tells us how to update $b_3$ to reduce the loss |

---

## 4. Computing Gradients for the Hidden Weights

### Gradient for $w_3$

The chain is longer now: $L \to \hat{y} \to z_3 \to h_1 \to z_1 \to w_1$

But the first two links are the same:

$$
\frac{\partial L}{\partial \hat{y}} = -2(y - \hat{y}), \quad \frac{\partial \hat{y}}{\partial z_3} = \sigma(z_3)(1 - \sigma(z_3))
$$

Then:

$$
\frac{\partial z_3}{\partial h_1} = w_3
$$

$$
\frac{\partial h_1}{\partial z_1} = \sigma(z_1)(1 - \sigma(z_1))
$$

$$
\frac{\partial z_1}{\partial w_1} = x
$$

Multiplying everything together:

$$
\frac{\partial L}{\partial w_1} = -2(y - \hat{y}) \cdot \sigma(z_3)(1 - \sigma(z_3)) \cdot w_3 \cdot \sigma(z_1)(1 - \sigma(z_1)) \cdot x
$$

### The Pattern

Every gradient follows the same structure:

$$
\frac{\partial L}{\partial w} = (\text{error signal}) \times (\text{path derivatives}) \times (\text{local input})
$$

The error signal starts at the loss. Every layer multiplies in its activation derivative and the weight connecting forward. By the time we reach the first layer, we have multiplied several terms together — but each term is a simple local derivative.

---

## 5. Updating Weights with Gradient Descent

Once we have all gradients, we update every parameter:

$$
w_1 \leftarrow w_1 - \alpha \cdot \frac{\partial L}{\partial w_1}
$$

$$
w_2 \leftarrow w_2 - \alpha \cdot \frac{\partial L}{\partial w_2}
$$

$$
w_3 \leftarrow w_3 - \alpha \cdot \frac{\partial L}{\partial w_3}
$$

$$
w_4 \leftarrow w_4 - \alpha \cdot \frac{\partial L}{\partial w_4}
$$

$$
b_1 \leftarrow b_1 - \alpha \cdot \frac{\partial L}{\partial b_1}, \quad b_2 \leftarrow b_2 - \alpha \cdot \frac{\partial L}{\partial b_2}, \quad b_3 \leftarrow b_3 - \alpha \cdot \frac{\partial L}{\partial b_3}
$$

Then repeat the forward pass, backward pass, and update for many iterations until the loss stops decreasing.

---

> **Check your intuition:** Why do we need all the intermediate values ($z_1, h_1, z_2, h_2, z_3, \hat{y}$) stored from the forward pass? What happens to the gradient for $w_1$ if $w_3$ is set to zero?

---

## Prerequisites and Further Reading

- **Previous:** L03 The Chain Rule, L04 Gradient Descent
- **Next:** L06 Backpropagation Details Part 1 — detailed derivation for multiple parameters
- **Related:** Gradient Descent lecture (the optimization engine that uses backpropagation's gradients)
