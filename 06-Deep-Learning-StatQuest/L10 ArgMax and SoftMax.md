## ArgMax and SoftMax

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. ArgMax

### Motivation and Intuition

After forward propagation, the raw output values (logits) can be any real number — greater than 1, less than 0, and hard to interpret. ArgMax solves this by making a hard decision: it sets the largest logit to 1 and all others to 0. This gives a clear "winner."

### ArgMax Formula

$$
\text{ArgMax}(z_i) =
\begin{cases}
1 & \text{if } z_i = \max(z_1, z_2, \dots, z_m) \\
0 & \text{otherwise}
\end{cases}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $z_i$ | Raw output (logit) for class $i$ | The input to the ArgMax function |
| $\text{ArgMax}(z_i)$ | 1 if $z_i$ is the largest, else 0 | Makes a discrete, interpretable prediction |

**Problem:** ArgMax has zero gradient almost everywhere and undefined gradient at the transition points — it cannot be used for backpropagation.

---

## 2. SoftMax

### Motivation and Intuition

SoftMax is a "soft" version of ArgMax that converts raw logits into a probability distribution (values between 0 and 1 that sum to 1). It preserves relative order while being smooth and differentiable, making it suitable for gradient-based training.

### SoftMax Formula

$$
\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{m} e^{z_j}} \quad \text{for } i = 1, \dots, m
$$

With temperature $T$:

$$
\sigma(z_i) = \frac{e^{z_i / T}}{\sum_{j=1}^{m} e^{z_j / T}}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $z_i$ | Raw logit for class $i$ | Input to SoftMax |
| $\sigma(z_i)$ | Predicted probability for class $i$ | Output between 0 and 1 |
| $T$ | Temperature parameter | $T > 1$ softens (more uniform), $T < 1$ sharpens (more confident) |
| $e^{z_i}$ | Exponentiated logit | Ensures positivity; large gaps become large ratios |

**Why SoftMax is differentiable:** Unlike ArgMax, the exponential function $e^{z_i}$ is smooth everywhere, and the division is a continuous differentiable operation. Every term in the denominator contributes to every output, creating rich gradients.

---

> **Check your intuition:** If logits are $[2.0, 1.0, 0.1]$, which class has the highest SoftMax probability? What happens if you set temperature $T=10$?

<details>
<summary>Answer</summary>
Class 0 has the highest probability. With $T=10$, the distribution becomes more uniform (softer).
</details>

---

## Prerequisites and Further Reading

- **Prerequisites:** L09 Multiple Inputs and Outputs.
- **Next:** L11 SoftMax Derivative Step by Step.
