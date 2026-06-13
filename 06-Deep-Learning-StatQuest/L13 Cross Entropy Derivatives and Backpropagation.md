## Cross Entropy Derivatives and Backpropagation

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. The Combined Cross-Entropy + SoftMax Derivative

### Motivation and Intuition

In practice, cross-entropy loss is almost always paired with a SoftMax output layer. Computing the derivative of cross-entropy with respect to the raw logits $\mathbf{z}$ (before SoftMax) combines two steps into one elegant formula. This combined derivative is remarkably simple: it is just the difference between the prediction and the target.

### Loss Function (Cross-Entropy)

$$
L = -\sum_{i=1}^{m} y_i \log(\hat{y}_i) \quad \text{where} \quad \hat{y}_i = \sigma(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

### Combined Derivative via the Chain Rule

We want $\frac{\partial L}{\partial z_k}$. Using the chain rule:

$$
\frac{\partial L}{\partial z_k} = \sum_{i=1}^{m} \frac{\partial L}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z_k}
$$

From cross-entropy: $\frac{\partial L}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i}$

From SoftMax Jacobian: $\frac{\partial \hat{y}_i}{\partial z_k} = \hat{y}_i (\delta_{ik} - \hat{y}_k)$ where $\delta_{ik} = 1$ if $i=k$ else $0$.

Putting them together:

$$
\frac{\partial L}{\partial z_k} = \sum_i -\frac{y_i}{\hat{y}_i} \cdot \hat{y}_i (\delta_{ik} - \hat{y}_k)
= -\sum_i y_i (\delta_{ik} - \hat{y}_k)
= -\left( y_k - \hat{y}_k \sum_i y_i \right)
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

### Why This Matters for Backpropagation

This simple gradient $\hat{y}_k - y_k$ flows backward through the network. The gradient is:
- **Positive** when the prediction exceeds the target (we need to reduce $z_k$).
- **Negative** when the prediction falls short (we need to increase $z_k$).
- **Zero** when the prediction is perfect.

This linear error signal propagates through the chain rule to update every weight and bias in earlier layers, making gradient flow strong and stable.

---

> **Check your intuition:** True class is setosa ($y = [1, 0, 0]$) and predicted probabilities are $\hat{y} = [0.8, 0.15, 0.05]$. What is the gradient $\partial L / \partial \mathbf{z}$?

<details>
<summary>Answer</summary>
$\partial L / \partial \mathbf{z} = [0.8 - 1, 0.15 - 0, 0.05 - 0] = [-0.2, 0.15, 0.05]$.
</details>

---

## Prerequisites and Further Reading

- **Prerequisites:** L11 SoftMax Derivative Step by Step, L12 Cross Entropy, L06-L07 Backpropagation Details.
- **Next:** L14 Convolutional Neural Networks.
