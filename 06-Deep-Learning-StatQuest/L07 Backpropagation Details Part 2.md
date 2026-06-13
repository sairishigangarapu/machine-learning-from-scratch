## L07 Backpropagation Details Part 2

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. Extending Backward Through the Hidden Layer

### Motivation and Intuition

In L06, we computed gradients for the output layer parameters $w_3, w_4, b_3$. Now we continue backward to compute gradients for the hidden layer parameters $w_1, b_1, w_2, b_2$. This is where the "chain rule goes bonkers" — the loss is connected to these early parameters through multiple intermediate steps.

The key insight: the error signal from the output gets **propagated backward** through the weights, scaled by the activation derivatives at each layer. The deeper the layer, the more terms in the chain, but each term is still just a simple local derivative.

### The Full Network

$$
z_1 = w_1 x + b_1, \quad h_1 = \sigma(z_1)
$$

$$
z_2 = w_2 x + b_2, \quad h_2 = \sigma(z_2)
$$

$$
z_3 = w_3 h_1 + w_4 h_2 + b_3, \quad \hat{y} = \sigma(z_3)
$$

$$
L = (y - \hat{y})^2
$$

All parameters are now unknown and must be optimized.

---

## 2. Gradients for $w_1$ and $b_1$ (Hidden Node 1)

### The Chain Rule Path for $w_1$

The path from $w_1$ to $L$: $w_1 \to z_1 \to h_1 \to z_3 \to \hat{y} \to L$

By the chain rule:

$$
\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_3} \cdot \frac{\partial z_3}{\partial h_1} \cdot \frac{\partial h_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}
$$

We already know the first two terms from the output error signal $\delta_3$:

$$
\delta_3 = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_3} = -2(y - \hat{y}) \cdot \sigma(z_3)(1 - \sigma(z_3))
$$

Now we need three more local derivatives:

$$
\frac{\partial z_3}{\partial h_1} = w_3
$$

$$
\frac{\partial h_1}{\partial z_1} = \sigma(z_1)(1 - \sigma(z_1))
$$

$$
\frac{\partial z_1}{\partial w_1} = x
$$

Multiply everything:

$$
\frac{\partial L}{\partial w_1} = \delta_3 \cdot w_3 \cdot \sigma(z_1)(1 - \sigma(z_1)) \cdot x
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\delta_3$ | Output error signal | The gradient of loss w.r.t. $z_3$ — computed once, reused for every parameter in earlier layers |
| $w_3$ | Weight from hidden node 1 to output | Propagates the error signal backward — large $w_3$ means hidden node 1 had more influence on the output, so it gets more responsibility for the error |
| $\sigma(z_1)(1 - \sigma(z_1))$ | Sigmoid derivative at hidden node 1 | If hidden node 1 is saturated (near 0 or 1), its derivative is near 0 and the gradient vanishes — the weight stops learning |
| $x$ | Input feature | The original input — scales the gradient for this weight |

### Gradient for $b_1$

The path is the same except the last step: $b_1 \to z_1$ instead of $w_1 \to z_1$

$$
\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_3} \cdot \frac{\partial z_3}{\partial h_1} \cdot \frac{\partial h_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial b_1}
$$

Since $\frac{\partial z_1}{\partial b_1} = 1$:

$$
\frac{\partial L}{\partial b_1} = \delta_3 \cdot w_3 \cdot \sigma(z_1)(1 - \sigma(z_1))
$$

---

## 3. Gradients for $w_2$ and $b_2$ (Hidden Node 2)

### Same Pattern, Different Branch

The path for $w_2$ follows the same structure but through hidden node 2:

$$
\frac{\partial L}{\partial w_2} = \delta_3 \cdot w_4 \cdot \sigma(z_2)(1 - \sigma(z_2)) \cdot x
$$

$$
\frac{\partial L}{\partial b_2} = \delta_3 \cdot w_4 \cdot \sigma(z_2)(1 - \sigma(z_2))
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $w_4$ | Weight from hidden node 2 to output | Same role as $w_3$ but for the second hidden node — error signal flows backward through this weight |
| $\sigma(z_2)(1 - \sigma(z_2))$ | Sigmoid derivative at hidden node 2 | Same vanishing gradient consideration as node 1 |
| $x$ | Input feature | Same input scales both $w_1$ and $w_2$ gradients |

---

## 4. The Complete Gradient Table

| Parameter | Gradient Formula |
| :--- | :--- |
| $b_3$ | $\delta_3$ |
| $w_3$ | $\delta_3 \cdot h_1$ |
| $w_4$ | $\delta_3 \cdot h_2$ |
| $b_1$ | $\delta_3 \cdot w_3 \cdot \sigma(z_1)(1 - \sigma(z_1))$ |
| $w_1$ | $\delta_3 \cdot w_3 \cdot \sigma(z_1)(1 - \sigma(z_1)) \cdot x$ |
| $b_2$ | $\delta_3 \cdot w_4 \cdot \sigma(z_2)(1 - \sigma(z_2))$ |
| $w_2$ | $\delta_3 \cdot w_4 \cdot \sigma(z_2)(1 - \sigma(z_2)) \cdot x$ |

Where $\delta_3 = -2(y - \hat{y}) \cdot \sigma(z_3)(1 - \sigma(z_3))$.

---

## 5. The Pattern: Error Signal Flows Backward

### The Key Insight for Any Network

For any layer in a neural network, the gradient follows this recipe:

1. **Start** with the error signal from the layer above.
2. **Multiply** by the weight connecting this node to the layer above (error flows backward through weights).
3. **Multiply** by the derivative of this node's activation function.
4. The result is the **error signal for this layer** — which you then pass further backward.
5. For a **weight**, also multiply by the input that fed into that weight (from the layer below).

### Reusable Error Signals

Define the error signal for hidden node 1:

$$
\delta_1 = \delta_3 \cdot w_3 \cdot \sigma(z_1)(1 - \sigma(z_1))
$$

Then all gradients for parameters feeding into hidden node 1 become simple:

$$
\frac{\partial L}{\partial w_1} = \delta_1 \cdot x, \quad \frac{\partial L}{\partial b_1} = \delta_1
$$

This modularity is what makes backpropagation so elegant. Error signals cascade backward, and each layer's gradients are a simple product of the incoming error signal and the local input.

---

## 6. The Full Algorithm

### One Training Iteration

**Forward Pass:** Compute $z_1, h_1, z_2, h_2, z_3, \hat{y}, L$. Store all intermediate values.

**Backward Pass:**
1. Compute $\delta_3 = -2(y - \hat{y}) \cdot \sigma(z_3)(1 - \sigma(z_3))$
2. Compute output gradients: $\frac{\partial L}{\partial w_3} = \delta_3 \cdot h_1$, $\frac{\partial L}{\partial w_4} = \delta_3 \cdot h_2$, $\frac{\partial L}{\partial b_3} = \delta_3$
3. Compute $\delta_1 = \delta_3 \cdot w_3 \cdot \sigma(z_1)(1 - \sigma(z_1))$, $\delta_2 = \delta_3 \cdot w_4 \cdot \sigma(z_2)(1 - \sigma(z_2))$
4. Compute hidden gradients: $\frac{\partial L}{\partial w_1} = \delta_1 \cdot x$, $\frac{\partial L}{\partial b_1} = \delta_1$, $\frac{\partial L}{\partial w_2} = \delta_2 \cdot x$, $\frac{\partial L}{\partial b_2} = \delta_2$

**Gradient Descent Update:** Update all 7 parameters using their gradients and learning rate $\alpha$.

**Repeat** for the next iteration.

---

> **Check your intuition:** If hidden node 1 is completely saturated (either $h_1 \approx 0$ or $h_1 \approx 1$), the derivative $\sigma(z_1)(1 - \sigma(z_1))$ is close to 0. What happens to the gradients for $w_1$ and $b_1$? What does this imply for the hidden node's ability to learn?

---

## Prerequisites and Further Reading

- **Previous:** L06 Backpropagation Details Part 1 (gradients for the output layer)
- **Next:** L08 ReLU in Action (a cure for the vanishing gradient problem)
- **Related:** L03 The Chain Rule, L04 Gradient Descent
