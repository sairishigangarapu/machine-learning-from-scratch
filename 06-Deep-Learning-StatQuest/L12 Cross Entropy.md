## Cross Entropy

*StatQuest with Josh Starmer — Clearly Explained*

---

## 1. Cross-Entropy Loss

### Motivation and Intuition

When a neural network outputs probabilities (via SoftMax), we need a loss function that measures how far the predicted distribution is from the true distribution. Mean Squared Error (MSE) works for regression but is suboptimal for classification because it treats probabilities as real numbers and ignores the probabilistic nature. Cross-entropy directly measures the "distance" between two probability distributions: the true labels (one-hot encoded) and the predicted probabilities.

### Cross-Entropy Formula for a Single Sample

For $m$ classes, with true labels one-hot encoded as $\mathbf{y} = [y_1, \dots, y_m]$ and predictions $\hat{\mathbf{y}} = [\hat{y}_1, \dots, \hat{y}_m]$:

$$
L = -\sum_{i=1}^{m} y_i \log(\hat{y}_i)
$$

Since $y_i$ is 1 for the correct class and 0 for all others, this simplifies to:

$$
L = -\log(\hat{y}_{\text{correct}})
$$

For a batch of $N$ samples:

$$
L = -\frac{1}{N} \sum_{j=1}^{N} \sum_{i=1}^{m} y_{ji} \log(\hat{y}_{ji})
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $y_i$ | True label for class $i$ (0 or 1) | One-hot encoded ground truth |
| $\hat{y}_i$ | Predicted probability for class $i$ | SoftMax output (between 0 and 1) |
| $L$ | Cross-entropy loss | Scalar measuring prediction error |
| $\log(\hat{y}_{\text{correct}})$ | Log-probability of the correct class | Drives loss toward 0 as prediction approaches 1 |

### Why Cross-Entropy Over MSE for Classification

1. **Gradient strength:** Cross-entropy penalizes confident wrong predictions much more heavily than MSE.
2. **Probabilistic interpretation:** Cross-entropy directly measures the number of bits needed to encode the true label using the predicted distribution.
3. **Faster convergence:** With SoftMax output, cross-entropy produces gradients proportional to $(\hat{y} - y)$, which is linear in the error. MSE produces vanishing gradients when predictions are near 0 or 1.

### Relationship to KL Divergence

Cross-entropy $H(P, Q)$ relates to KL divergence:

$$
H(P, Q) = H(P) + D_{KL}(P \parallel Q)
$$

where $H(P)$ is the entropy of the true distribution (constant for fixed labels, so minimizing cross-entropy is equivalent to minimizing KL divergence).

---

> **Check your intuition:** If the true class is class 2 (index 1) and the predicted probabilities are $[0.1, 0.8, 0.1]$, what is the cross-entropy loss?

<details>
<summary>Answer</summary>
$L = -\log(0.8) \approx 0.223$.
</details>

---

## Prerequisites and Further Reading

- **Prerequisites:** L10 ArgMax and SoftMax, L02 Neural Networks Part 1 (backpropagation ideas).
- **Next:** L13 Cross Entropy Derivatives and Backpropagation.
