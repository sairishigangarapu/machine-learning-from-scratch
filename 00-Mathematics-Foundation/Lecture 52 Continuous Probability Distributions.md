## Continuous Probability Distributions

*Essential Mathematics for ML — Structured Notes*

---

## 1. The Normal (Gaussian) Distribution

### Motivation and Intuition
The Gaussian is the most important distribution in all of machine learning. The Central Limit Theorem guarantees that sums of independent random variables converge to a Gaussian. Weight initializations, noise models, variational autoencoders, and kernel methods all assume Gaussianity.

### Definition
$$
X \sim \mathcal{N}(\mu, \sigma^2): \quad f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

### Standard Normal
$$
Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)
$$

### The 68-95-99.7 Rule
$$
P(\mu - \sigma < X < \mu + \sigma) \approx 0.68
$$
$$
P(\mu - 2\sigma < X < \mu + 2\sigma) \approx 0.95
$$
$$
P(\mu - 3\sigma < X < \mu + 3\sigma) \approx 0.997
$$

```python
from scipy.stats import norm

# Standard normal probabilities
print(f"P(-1 < Z < 1) = {norm.cdf(1) - norm.cdf(-1):.4f}")  # 0.6827
print(f"P(-2 < Z < 2) = {norm.cdf(2) - norm.cdf(-2):.4f}")  # 0.9545
print(f"P(-3 < Z < 3) = {norm.cdf(3) - norm.cdf(-3):.4f}")  # 0.9973
```

### Multivariate Gaussian
$$
\mathcal{N}(\boldsymbol{\mu}, \Sigma): \quad f(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^d |\Sigma|}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)
$$

Used in: Gaussian Mixture Models, Kalman filters, Gaussian Processes.

```python
import numpy as np
from scipy.stats import multivariate_normal

mu = np.array([0, 0])
Sigma = np.array([[1, 0.5], [0.5, 1]])

rv = multivariate_normal(mean=mu, cov=Sigma)
print(f"P([0.5, 0.5]) = {rv.pdf([0.5, 0.5]):.4f}")
```

---

## 2. The Exponential Distribution

$$
f(x) = \lambda e^{-\lambda x}, \quad x \ge 0
$$

* Mean: $E[X] = 1/\lambda$
* Variance: $\text{Var}(X) = 1/\lambda^2$

**Memoryless property:** $P(X > s + t | X > s) = P(X > t)$

**ML Connection:** Models waiting times, time between events. Used in survival analysis and reliability engineering.

```python
from scipy.stats import expon

# Mean time between events with rate lambda = 2
print(f"E[X] = {expon.mean(scale=0.5):.2f}")  # 0.5
print(f"P(X < 1) = {expon.cdf(1, scale=0.5):.4f}")
```

---

## 3. The Uniform Distribution

$$
f(x) = \frac{1}{b - a}, \quad a \le x \le b
$$

**ML Connection:** Random weight initialization often uses uniform distributions. Xavier initialization samples from $\mathcal{U}(-\frac{1}{\sqrt{n}}, \frac{1}{\sqrt{n}})$.

---

## 4. The Beta Distribution

$$
f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad x \in [0, 1]
$$

**ML Connection:** Prior distribution for probabilities in Bayesian inference. The posterior of a Beta prior with Binomial likelihood is also Beta (conjugate prior).

| $\alpha, \beta$ | Shape |
|:---|:---|
| $\alpha = \beta = 1$ | Uniform |
| $\alpha = \beta > 1$ | Symmetric, peaked at 0.5 |
| $\alpha > \beta$ | Skewed right |
| $\alpha < \beta$ | Skewed left |

---

## 5. The Laplace Distribution

$$
f(x) = \frac{1}{2b} \exp\left(-\frac{|x - \mu|}{b}\right)
$$

**ML Connection:** The Laplace distribution is the prior that produces **L1 regularization** (Lasso). Maximizing the posterior with a Laplace prior is equivalent to minimizing the loss plus an L1 penalty on the weights.

---

## 6. Choosing the Right Distribution

| If your data... | Use distribution |
|:---|:---|
| Is symmetric, bell-shaped | Gaussian |
| Is skewed, non-negative | Exponential, Log-normal |
| Represents a probability | Beta |
| Has heavy tails | Student-t |
| Counts events | Poisson |
| Is binary (0/1) | Bernoulli/Binomial |

> **Check your intuition:** Why does adding many small independent noise sources produce a Gaussian distribution? *(Answer: The Central Limit Theorem. Regardless of the individual distributions, the sum of many independent random variables converges to a Gaussian as the number of variables grows.)*
