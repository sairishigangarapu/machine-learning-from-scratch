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

| Term | Definition | Significance |
|:---|:---|:---|
| $E[X]$ | **Expected value** (mean) of $X$ | The center of mass of a distribution; in ML, we minimize expected loss $\mathbb{E}[\ell(f(\mathbf{x}), y)]$ over the data distribution |
| $x$ | Possible value of the random variable | Each outcome $x$ contributes to the average weighted by its probability |
| $p(x)$ | **Probability Mass Function** — $P(X = x)$ | Weights each outcome by how likely it is; more probable outcomes dominate the expectation |
| $\sum_x$ | Sum over all possible values of $X$ | For discrete variables, expectation is a weighted sum |

### Continuous Case
$$
E[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $E[X]$ | **Expected value** (mean) of continuous random variable $X$ | The center of mass of a continuous distribution; in ML, we minimize the expected loss $\mathbb{E}[\ell(f(\mathbf{x}), y)]$ over the data distribution |
| $x$ | Value of the continuous random variable | Integrated over all possible real numbers weighted by the density |
| $f(x)$ | **Probability Density Function** (PDF) of $X$ | Describes the relative likelihood of $X$ taking values near $x$ |
| $\int_{-\infty}^{\infty}$ | Integral over the entire real line | For continuous variables, expectation is an integral rather than a sum |

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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\text{Var}(X)$ | **Variance** — expected squared deviation from the mean | Measures the spread/dispersion of a distribution; fundamental for understanding model uncertainty and the bias-variance tradeoff in ML |
| $\mu$ | Mean $E[X]$ | The center point around which variance is measured |
| $E[(X - \mu)^2]$ | Definitional formula for variance | Directly captures the average squared distance from the mean |
| $E[X^2] - (E[X])^2$ | Computational formula for variance | Often easier to compute in practice; derived by expanding the square inside the expectation |

### Standard Deviation
$$
\sigma = \sqrt{\text{Var}(X)}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\sigma$ | **Standard deviation** — square root of the variance | Measures spread in the same units as $X$; $\sigma$ is the natural scale for confidence intervals, normalization (z-scores), and the 68-95-99.7 rule |
| $\text{Var}(X)$ | Variance of $X$ | The squared standard deviation |
| $\sqrt{\cdot}$ | Square root operation | Converts variance back to the original units of $X$ |

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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\text{Cov}(X, Y)$ | **Covariance** — measure of joint variability between $X$ and $Y$ | Positive: $X$ and $Y$ move together; negative: they move oppositely; zero: uncorrelated. Essential for PCA and understanding feature relationships |
| $E[XY]$ | Expected value of the product $XY$ | Joint moment that captures the product of deviations |
| $E[X]E[Y]$ | Product of individual expectations | Subtracted to center the covariance; for independent variables, $\text{Cov}(X, Y) = 0$ |

* Positive: variables move together. Negative: variables move oppositely. Zero: uncorrelated.

### Covariance Matrix
For a random vector $\mathbf{X} \in \mathbb{R}^n$:

$$
\Sigma = \text{Cov}(\mathbf{X}) = E[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T]
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\Sigma$ | **Covariance matrix** of random vector $\mathbf{X}$ | The matrix at the heart of PCA, Mahalanobis distance, and multivariate Gaussian distributions; captures both variances (diagonal) and covariances (off-diagonal) |
| $\mathbf{X}$ | Random vector in $\mathbb{R}^n$ | A collection of $n$ random variables, each representing a feature |
| $\boldsymbol{\mu}$ | Mean vector $E[\mathbf{X}]$ | The vector of individual feature means |
| $(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T$ | Outer product of deviation vectors | Produces an $n \times n$ matrix; the $(i,j)$ entry is $\text{Cov}(X_i, X_j)$ |
| $E[\cdot]$ | Expectation of a matrix | Applied element-wise to yield the covariance matrix |

This is the matrix at the heart of PCA (Lecture 16).

### Correlation Coefficient
$$
\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\rho_{XY}$ | **Correlation coefficient** — normalized covariance between $X$ and $Y$ | Scale-invariant measure of linear dependence; always in $[-1, 1]$; $\rho = \pm 1$ means perfect linear relationship |
| $\text{Cov}(X, Y)$ | Covariance between $X$ and $Y$ | Unnormalized measure of joint variability |
| $\sigma_X$ | Standard deviation of $X$ | Normalizes $X$'s contribution to unit scale |
| $\sigma_Y$ | Standard deviation of $Y$ | Normalizes $Y$'s contribution to unit scale |

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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $P(|X - \mu| \ge k\sigma)$ | Probability that $X$ deviates from its mean by at least $k$ standard deviations | The quantity bounded by Chebyshev's inequality; used in ML for proving generalization bounds via concentration inequalities |
| $k$ | Number of standard deviations | Positive real number; larger $k$ gives a tighter upper bound |
| $\frac{1}{k^2}$ | Chebyshev bound | Universal upper bound valid for ANY distribution with finite variance; $k=2$ gives $\le 0.25$, $k=3$ gives $\le 0.11$ |
| $\mu$ | Mean of $X$ | Center of the distribution |
| $\sigma$ | Standard deviation of $X$ | Scale of the distribution |

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

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 49: Bayes Theorem and Random Variables](Lecture%2049%20Bayes%20Theorem%20and%20Random%20Variables.md) — Defines random variables and distributions
- **Next:** [Lecture 51: PMF and PDF](Lecture%2051%20PMF%20and%20PDF.md) — Formalizes distribution functions
- **Related:** [Lecture 49: Bayes Theorem and Random Variables](Lecture%2049%20Bayes%20Theorem%20and%20Random%20Variables.md) — Context for expectation calculations
- **Related:** [Lecture 52: Continuous Probability Distributions](Lecture%2052%20Continuous%20Probability%20Distributions.md) — Applies expectation to continuous distributions
