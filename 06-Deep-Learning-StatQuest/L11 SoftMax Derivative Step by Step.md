## SoftMax Derivative Step by Step

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. Derivative of SoftMax

### Motivation and Intuition

To train a neural network with SoftMax output using gradient descent, we need the derivative of the SoftMax function with respect to each raw logit $z_k$. Because each SoftMax output $\sigma_i$ depends on ALL logits (through the denominator), the derivative has two cases: the diagonal case ($i = k$) and the off-diagonal case ($i \neq k$). The result is a Jacobian matrix.

### SoftMax Recap

$$
\sigma_i = \frac{e^{z_i}}{\sum_{j=1}^{m} e^{z_j}}
$$

### Derivative: Diagonal Case ($i = k$)

Using the quotient rule:

$$
\frac{\partial \sigma_i}{\partial z_i} = \frac{e^{z_i} \cdot \sum_{j} e^{z_j} - e^{z_i} \cdot e^{z_i}}{\left(\sum_{j} e^{z_j}\right)^2}
= \frac{e^{z_i}}{\sum_{j} e^{z_j}} \cdot \frac{\sum_{j} e^{z_j} - e^{z_i}}{\sum_{j} e^{z_j}}
= \sigma_i (1 - \sigma_i)
$$

### Derivative: Off-Diagonal Case ($i \neq k$)

$$
\frac{\partial \sigma_i}{\partial z_k} = \frac{0 \cdot \sum_{j} e^{z_j} - e^{z_i} \cdot e^{z_k}}{\left(\sum_{j} e^{z_j}\right)^2}
= - \frac{e^{z_i}}{\sum_{j} e^{z_j}} \cdot \frac{e^{z_k}}{\sum_{j} e^{z_j}}
= - \sigma_i \sigma_k
$$

### Jacobian Matrix

The full Jacobian $J \in \mathbb{R}^{m \times m}$ has entries:

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
| $\sigma_i$ | SoftMax probability for class $i$ | Output of SoftMax |
| $z_k$ | Raw logit for class $k$ | Input to SoftMax |
| $\frac{\partial \sigma_i}{\partial z_i}$ | Derivative w.r.t. same logit | Always positive; max at $\sigma_i = 0.5$ |
| $\frac{\partial \sigma_i}{\partial z_k}$ | Derivative w.r.t. different logit | Always negative; increasing $z_k$ decreases $\sigma_i$ |
| $J$ | Jacobian matrix, $m \times m$ | Full derivative of SoftMax w.r.t. all logits |

---

> **Check your intuition:** If $\sigma = [0.7, 0.2, 0.1]$, what is the Jacobian entry $J_{11}$? What about $J_{12}$?

<details>
<summary>Answer</summary>
$J_{11} = 0.7 \times (1 - 0.7) = 0.21$. $J_{12} = -0.7 \times 0.2 = -0.14$.
</details>

---

## Prerequisites and Further Reading

- **Prerequisites:** L10 ArgMax and SoftMax, quotient rule, partial derivatives.
- **Next:** L12 Cross Entropy.
