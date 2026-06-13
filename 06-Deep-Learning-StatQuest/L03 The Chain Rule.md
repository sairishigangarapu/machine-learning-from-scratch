## L03 The Chain Rule

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. What the Chain Rule Is

### Motivation and Intuition

The chain rule is a calculus technique for computing the derivative of a **composite function** — a function inside another function. In neural networks, the output is the result of many nested operations: you multiply by a weight, add a bias, run through an activation, multiply by another weight, and so on. To figure out how much each weight contributed to the final error, you need to work backward through these nested operations. The chain rule is the tool that makes this possible.

Think of it like a factory assembly line. If the final product is defective, you trace backward: the last station feeds into quality control, the second-to-last feeds into the last, and so on. The chain rule lets you measure how much each station contributed to the defect, given how each station affects the next.

### Intuitive Statement

If $y = f(u)$ and $u = g(x)$, then the derivative of $y$ with respect to $x$ is:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

In plain English: "How much does $y$ change when $x$ changes?" = "How much does $y$ change when $u$ changes?" times "How much does $u$ change when $x$ changes?"

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\frac{dy}{dx}$ | Derivative of $y$ w.r.t. $x$ | The quantity we want — how the final output responds to the original input |
| $\frac{dy}{du}$ | Derivative of outer function w.r.t. inner variable | How the output changes when the intermediate value changes |
| $\frac{du}{dx}$ | Derivative of inner function w.r.t. input | How the intermediate value changes when the input changes |
| $f(u)$ | Outer function | Applied after the inner transformation — e.g., sigmoid applied to the weighted sum |
| $g(x)$ | Inner function | The first transformation — e.g., the weighted sum $wx + b$ |

---

## 2. Single-Variable Chain Rule

### Step-by-Step Example

Suppose we have:

$$
y = \sigma(z) \quad\text{where}\quad z = wx + b
$$

We want $\frac{dy}{dw}$ — how does the prediction change when we nudge the weight?

By the chain rule:

$$
\frac{dy}{dw} = \frac{dy}{dz} \cdot \frac{dz}{dw}
$$

Breaking it down:

$$
\frac{dy}{dz} = \sigma'(z) = \sigma(z)(1 - \sigma(z)) \quad\text{(derivative of sigmoid)}
$$

$$
\frac{dz}{dw} = x \quad\text{(derivative of $wx + b$ w.r.t. $w$)}
$$

Therefore:

$$
\frac{dy}{dw} = \sigma(z)(1 - \sigma(z)) \cdot x
$$

### Why This Matters for Neural Networks

Every weight in the network sits at the end of a chain of operations. The chain rule lets us compute the gradient of the loss with respect to that weight by multiplying a series of local derivatives together. No other technique gives us this clean, modular decomposition.

---

## 3. Multivariable Chain Rule

### Multiple Inputs, One Output

In neural networks, most functions have multiple inputs. For example, a hidden node receives multiple weighted inputs. The multivariable chain rule handles this case.

If $z = w_1 x_1 + w_2 x_2 + b$, and $y = \sigma(z)$, then:

$$
\frac{\partial y}{\partial w_1} = \frac{dy}{dz} \cdot \frac{\partial z}{\partial w_1} = \sigma'(z) \cdot x_1
$$

$$
\frac{\partial y}{\partial w_2} = \frac{dy}{dz} \cdot \frac{\partial z}{\partial w_2} = \sigma'(z) \cdot x_2
$$

Notice the switch from $d$ (total derivative) to $\partial$ (partial derivative). A partial derivative means "how does this one input affect the output, holding everything else constant."

### General Form for a Multi-Layer Network

For a loss $L$ that depends on a weight $w$ somewhere deep in the network:

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h_k} \cdot \frac{\partial h_k}{\partial h_{k-1}} \cdot \dots \cdot \frac{\partial h_1}{\partial w}
$$

Each term in this product is a **local gradient** — the derivative of one operation with respect to its immediate input. Multiply them all together, and you get the gradient of the loss with respect to any weight in the network.

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\frac{\partial L}{\partial \hat{y}}$ | Gradient of loss w.r.t. prediction | How much the error changes when the final prediction changes — the "error signal" |
| $\frac{\partial \hat{y}}{\partial h_k}$ | Gradient of prediction w.r.t. last hidden layer | How the prediction responds to the final hidden representation |
| $\frac{\partial h_{k}}{\partial h_{k-1}}$ | Gradient through hidden layer $k$ | How activation $k$ changes when the previous layer's activations change |
| $\frac{\partial h_1}{\partial w}$ | Gradient of first hidden activation w.r.t. weight | How the first hidden node responds to a specific weight |

---

## 4. Visualizing the Chain

### The Chain as a Path

Imagine a computational graph. Each node is an operation, and each edge shows the flow of data. The chain rule says: to find the gradient of the loss with respect to a weight, trace every path from that weight to the loss, multiply the derivatives along each path, and sum across paths.

```
w₁ ──→ z₁ ──→ h₁ ──→ z₃ ──→ ŷ ──→ L
                    ↑
w₂ ──→ z₂ ──→ h₂ ──┘
```

The gradient $\frac{\partial L}{\partial w_1}$ follows the path: $L \to \hat{y} \to z_3 \to h_1 \to z_1 \to w_1$, multiplying the local derivative at each step.

---

> **Check your intuition:** If $y = \sigma(wx + b)$ and you know $\frac{dy}{dz} = \sigma(z)(1-\sigma(z))$ where $z = wx + b$, what is $\frac{dy}{db}$?

---

## Prerequisites and Further Reading

- **Previous:** L02 Neural Networks Part 1 — Essential Main Ideas
- **Next:** L04 Gradient Descent (what we do with these gradients)
- **Related:** Differential Calculus — derivatives of elementary functions
