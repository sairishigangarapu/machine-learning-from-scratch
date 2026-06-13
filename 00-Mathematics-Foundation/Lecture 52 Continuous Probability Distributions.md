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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $X$ | Random variable | Observation drawn from the distribution |
| $\mu$ | Mean (location) parameter | Center of the distribution; also the median and mode |
| $\sigma^2$ | Variance (scale) parameter | Spread of the distribution around the mean |
| $\sigma$ | Standard deviation | $\sqrt{\sigma^2}$, controls the width of the bell curve |
| $f(x)$ | Probability density function (PDF) | Height of the curve at point $x$; not a probability itself |

### Standard Normal
$$
Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $Z$ | Standard normal variable | Number of standard deviations from the mean |
| $X$ | Original random variable | Raw observation to be standardized |
| $\mu$ | Mean of $X$ | Centering offset for standardization |
| $\sigma$ | Standard deviation of $X$ | Scaling factor for standardization |
| $\mathcal{N}(0, 1)$ | Standard normal distribution | Zero-mean, unit-variance Gaussian used for lookup tables |

### The 68-95-99.7 Rule
$$
\begin{aligned}
P(\mu - \sigma < X < \mu + \sigma) &\approx 0.68 \\
P(\mu - 2\sigma < X < \mu + 2\sigma) &\approx 0.95 \\
P(\mu - 3\sigma < X < \mu + 3\sigma) &\approx 0.997
\end{aligned}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mu \pm \sigma$ | One standard deviation from the mean | 68% of data lies within 1$\sigma$ — the most common interval |
| $\mu \pm 2\sigma$ | Two standard deviations | 95% coverage — often used for statistical significance thresholds |
| $\mu \pm 3\sigma$ | Three standard deviations | 99.7% coverage — points beyond are considered outliers |
| $P(a < X < b)$ | Probability that $X$ lies in the interval $(a, b)$ | Defines the area under the Normal PDF between the bounds |

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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\mathbf{x}$ | $d$-dimensional random vector | Multivariate observation |
| $\boldsymbol{\mu}$ | Mean vector | Center of the distribution in $\mathbb{R}^d$ |
| $\Sigma$ | Covariance matrix | Captures pairwise feature correlations and scales |
| $d$ | Dimensionality | Number of components in the random vector |
| $|\Sigma|$ | Determinant of $\Sigma$ | Normalizing constant for the PDF |
| $\Sigma^{-1}$ | Inverse covariance matrix | Measures quadratic form (Mahalanobis distance) |

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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $x$ | Non-negative random variable | Waiting time or inter-arrival time |
| $\lambda$ | Rate parameter | Average number of events per unit time |
| $f(x)$ | Probability density function | Decays exponentially with $x$ |

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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $x$ | Random variable on $[a, b]$ | Observation from the interval |
| $a$ | Lower bound | Minimum possible value |
| $b$ | Upper bound | Maximum possible value |
| $f(x)$ | Probability density function | Constant over the support $[a, b]$ |

**ML Connection:** Random weight initialization often uses uniform distributions. Xavier initialization samples from $\mathcal{U}(-\frac{1}{\sqrt{n}}, \frac{1}{\sqrt{n}})$.

---

## 4. The Beta Distribution

$$
f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad x \in [0, 1]
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\alpha, \beta$ | Shape parameters | Control the distribution's skew and concentration |
| $x^{\alpha-1}(1-x)^{\beta-1}$ | Kernel of the Beta distribution | Unnormalized density — captures the shape before scaling |
| $B(\alpha, \beta)$ | Beta function (normalization constant) | Ensures the PDF integrates to 1 over $[0,1]$ |
| $x \in [0, 1]$ | Support | Beta is defined only on the unit interval — ideal for probability modeling |

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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $x$ | Random variable | Observation from the distribution |
| $\mu$ | Location parameter | Center (median and mode) of the distribution |
| $b$ | Scale parameter | Controls spread; larger $b$ gives heavier tails |
| $f(x)$ | Probability density function | Symmetric, exponential decay in both directions |

**ML Connection:** The Laplace distribution is the prior that produces **L1 regularization** (Lasso). Maximizing the posterior with a Laplace prior is equivalent to minimizing the loss plus an L1 penalty on the weights.

---

## 6. The Gamma Distribution

### Motivation and Intuition
The Gamma distribution generalizes the Exponential distribution. While the Exponential models waiting time for *one* event, the Gamma models waiting time for *k* events. It's the natural prior for positive-valued parameters in Bayesian models.

### Definition
$$
f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $x$ | Positive random variable | Waiting time for $\alpha$ events |
| $\alpha$ | Shape parameter | Number of events; controls the curve's shape |
| $\beta$ | Rate parameter | Inverse scale; event occurrence rate |
| $\Gamma(\alpha)$ | Gamma function | Normalizing constant; $\Gamma(\alpha) = (\alpha-1)!$ for integer $\alpha$ |
| $f(x)$ | Probability density function | Generalizes the Exponential distribution |

where $\Gamma(\alpha) = \int_0^\infty t^{\alpha-1} e^{-t} \, dt$ is the Gamma function.

* Mean: $E[X] = \alpha / \beta$
* Variance: $\text{Var}(X) = \alpha / \beta^2$

### Special Cases

| Parameters | Distribution |
|:---|:---|
| $\alpha = 1$ | Exponential($\beta$) |
| $\alpha = n/2, \beta = 1/2$ | Chi-squared($n$) |
| $\alpha$ large | Approaches Gaussian |

**ML Connection:** Gamma priors for variance parameters in Bayesian models. The Chi-squared distribution (special case) appears in feature selection and goodness-of-fit tests.

```python
from scipy.stats import gamma
import numpy as np

# Gamma(α=2, β=1)
print(f"E[X] = {gamma.mean(a=2, scale=1):.2f}")  # 2.0
print(f"Var(X) = {gamma.var(a=2, scale=1):.2f}")  # 2.0
print(f"P(X < 3) = {gamma.cdf(3, a=2, scale=1):.4f}")
```

---

## 7. The Student-t Distribution

### Motivation and Intuition
The Student-t distribution has **heavier tails** than the Gaussian — it assigns more probability to extreme values. It arises when estimating the mean of a normally distributed population with unknown variance and small sample size. In ML, it's the foundation of robust regression and Bayesian inference.

### Definition
$$
f(x) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{x^2}{\nu}\right)^{-(\nu+1)/2}
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $x$ | Random variable | Observation centered at 0 |
| $\nu$ | Degrees of freedom | Controls tail heaviness; small $\nu$ = heavier tails |
| $\Gamma(\cdot)$ | Gamma function | Normalizing constant in the PDF |
| $f(x)$ | Probability density function | Bell-shaped with heavier tails than Gaussian |

where $\nu$ is the **degrees of freedom**.

* As $\nu \to \infty$: approaches $\mathcal{N}(0, 1)$
* Small $\nu$: heavy tails, more outliers

### Properties

| $\nu$ | Tail weight | Use case |
|:---|:---|:---|
| 1 | Very heavy (Cauchy) | Rare, extreme events |
| 5 | Heavy | Small-sample robust inference |
| 30 | Nearly Gaussian | Large-sample approximation |

**ML Connection:** Robust regression replaces Gaussian noise with Student-t to downweight outliers. Bayesian linear regression with unknown variance uses the Student-t posterior.

```python
from scipy.stats import t as student_t
import numpy as np

# Compare tails: Student-t(ν=5) vs Gaussian
print(f"P(|X| > 3) for Gaussian:  {2 * (1 - student_t.cdf(3, df=100)):.4f}")
print(f"P(|X| > 3) for t(ν=5):    {2 * (1 - student_t.cdf(3, df=5)):.4f}")
# t-distribution assigns more probability to extreme values
```

---

## 8. Choosing the Right Distribution

| If your data... | Use distribution |
|:---|:---|
| Is symmetric, bell-shaped | Gaussian |
| Is skewed, non-negative | Exponential, Gamma |
| Represents a probability | Beta |
| Has heavy tails | Student-t |
| Counts events | Poisson |
| Is binary (0/1) | Bernoulli/Binomial |
| Waiting time for k events | Gamma |

> **Check your intuition:** Why does adding many small independent noise sources produce a Gaussian distribution? *(Answer: The Central Limit Theorem. Regardless of the individual distributions, the sum of many independent random variables converges to a Gaussian as the number of variables grows.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 51: PMF and PDF](Lecture%2051%20PMF%20and%20PDF.md) — General distribution functions
- **Next:** [Lecture 53: Joint Probability Distributions](Lecture%2053%20Joint%20Probability%20Distributions.md) — Extensions to multiple random variables
- **Related:** [Lecture 49: Bayes Theorem and Random Variables](Lecture%2049%20Bayes%20Theorem%20and%20Random%20Variables.md) — Introduces continuous distributions
- **Related:** [Lecture 50: Expectation and Variance](Lecture%2050%20Expectation%20and%20Variance.md) — Statistical properties of these distributions
- **Related:** [Lecture 51: PMF and PDF](Lecture%2051%20PMF%20and%20PDF.md) — Mathematical foundation for these distributions
