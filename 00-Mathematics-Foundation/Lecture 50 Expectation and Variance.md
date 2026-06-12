## Expectation and Variance

*Essential Mathematics for ML — Structured Notes*

---

## 1. Expected Value (Mean)

### Motivation and Intuition
The **expected value** is the "center of mass" of a probability distribution. In ML, we minimize the *expected* loss over the data distribution, not the loss on individual samples.

### Discrete Case
$$
E[X] = \sum_x x \cdot p(x)
$$

### Continuous Case
$$
E[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx
$$

### Properties

| Property | Formula |
|:---|:---|
| **Linearity** | $E[aX + b] = aE[X] + b$ |
| **Sum** | $E[X + Y] = E[X] + E[Y]$ (always, even if dependent) |
| **Product** | $E[XY] = E[X]E[Y]$ only if $X, Y$ are independent |

```python
import numpy as np
from scipy.stats import uniform, norm

outcomes = np.array([1, 2, 3, 4, 5, 6])
probs = np.ones(6) / 6
print(f"E[die] = {np.sum(outcomes * probs)}")  # 3.5
print(f"E[Uniform(0,1)] = {uniform.mean(loc=0, scale=1)}")  # 0.5
```

---

## 2. Variance

### Definition
Measures spread around the mean:

$$
\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2
$$

### Standard Deviation
$$
\sigma = \sqrt{\text{Var}(X)}
$$

### Properties

| Property | Formula |
|:---|:---|
| **Constant** | $\text{Var}(c) = 0$ |
| **Scaling** | $\text{Var}(aX) = a^2 \text{Var}(X)$ |
| **Independence** | $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ if independent |
| **General** | $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$ |

```python
print(f"Var[N(0,1)] = {norm.var(loc=0, scale=1)}")   # 1.0
print(f"Var[N(3,4)] = {norm.var(loc=3, scale=2)}")   # 4.0
```

---

## 3. Covariance and Correlation

### Covariance
$$
\text{Cov}(X, Y) = E[XY] - E[X]E[Y]
$$

* Positive: variables move together. Negative: variables move oppositely. Zero: uncorrelated.

### Covariance Matrix
For a random vector $\mathbf{X} \in \mathbb{R}^n$:

$$
\Sigma = \text{Cov}(\mathbf{X}) = E[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T]
$$

This is the matrix at the heart of PCA (Lecture 16).

### Correlation Coefficient
$$
\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]
$$

```python
import numpy as np

X = np.random.randn(1000)
Y = 2 * X + 0.5 * np.random.randn(1000)

cov_matrix = np.cov(X, Y)
print(f"Cov(X,Y) = {cov_matrix[0,1]:.4f}")  # ~1.0

corr = np.corrcoef(X, Y)[0, 1]
print(f"Corr(X,Y) = {corr:.4f}")  # ~0.97
```

---

## 4. Moments

The $k$-th **moment** of $X$ is $E[X^k]$. The $k$-th **central moment** is $E[(X - \mu)^k]$.

| Moment | Central Moment | Name |
|:---|:---|:---|
| $E[X]$ | — | Mean |
| $E[X^2]$ | $\text{Var}(X)$ | Variance |
| $E[X^3]$ | Skewness (normalized) | Asymmetry |
| $E[X^4]$ | Kurtosis (normalized) | Tail weight |

**ML Connection:** Skewness and kurtosis of feature distributions inform data preprocessing. Highly skewed features benefit from log transforms. Heavy-tailed distributions require robust loss functions.

---

## 5. Chebyshev's Inequality

For any random variable with mean $\mu$ and variance $\sigma^2$:

$$
P(|X - \mu| \ge k\sigma) \le \frac{1}{k^2}
$$

This gives a worst-case bound on how far $X$ can be from its mean.

**ML Connection:** Provides theoretical guarantees on generalization error. If the training loss has bounded variance, Chebyshev's inequality bounds the probability of large deviations.

---

## 6. Why Expectation and Variance Matter in ML

| Concept | Role |
|:---|:---|
| **Expected risk** | $R(f) = E[\ell(f(\mathbf{x}), y)]$ — what we actually minimize |
| **Empirical risk** | $\hat{R}(f) = \frac{1}{n}\sum \ell(f(\mathbf{x}_i), y_i)$ — approximation |
| **Variance of estimator** | How much the model changes with different training sets |
| **Bias-variance tradeoff** | Decomposition of expected error into bias and variance |

> **Check your intuition:** If $X$ and $Y$ are independent with $\text{Var}(X) = 3$ and $\text{Var}(Y) = 5$, what is $\text{Var}(X + Y)$? *(Answer: $3 + 5 = 8$. Independence means the covariance is zero.)*
