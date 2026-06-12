## PMF and PDF

*Essential Mathematics for ML — Structured Notes*

---

## 1. Probability Mass Function (PMF)

### Discrete Random Variables
A discrete random variable $X$ takes countable values. The **PMF** gives the probability at each point:

$$
p(x) = P(X = x)
$$

**Properties:**
* $p(x) \ge 0$ for all $x$
* $\sum_x p(x) = 1$

### Example: Fair Die
$$
p(x) = \frac{1}{6}, \quad x \in \{1, 2, 3, 4, 5, 6\}
$$

```python
import numpy as np
from scipy.stats import binom

# Binomial PMF: P(X = k) = C(n,k) p^k (1-p)^(n-k)
n, p = 10, 0.3
x = np.arange(0, 11)
pmf = binom.pmf(x, n, p)

for xi, pi in zip(x, pmf):
    print(f"P(X={xi:2d}) = {pi:.4f}")
```

---

## 2. Probability Density Function (PDF)

### Continuous Random Variables
For continuous $X$, $P(X = x) = 0$ for any specific point. Instead, we use the **PDF**:

$$
P(a \le X \le b) = \int_a^b f(x) \, dx
$$

**Properties:**
* $f(x) \ge 0$
* $\int_{-\infty}^{\infty} f(x) \, dx = 1$

Note: $f(x)$ can be greater than 1 — it's a density, not a probability.

### Relationship to CDF
$$
f(x) = \frac{dF(x)}{dx}, \quad F(x) = \int_{-\infty}^{x} f(t) \, dt
$$

```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 200)
pdf = norm.pdf(x, loc=0, scale=1)
cdf = norm.cdf(x, loc=0, scale=1)

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(x, pdf, 'b-', lw=2)
plt.title("PDF: f(x)")
plt.subplot(122)
plt.plot(x, cdf, 'r-', lw=2)
plt.title("CDF: F(x)")
plt.tight_layout()
```

---

## 3. Key Distributions

### Normal (Gaussian) Distribution
$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

The most important distribution in ML. Appears in:
* Feature modeling (Gaussian Naive Bayes)
* Weight initialization (Xavier/He)
* Variational Autoencoders (VAE)
* Gaussian Processes

### Exponential Distribution
$$
f(x) = \lambda e^{-\lambda x}, \quad x \ge 0
$$

Used in: survival analysis, modeling time between events.

### Uniform Distribution
$$
f(x) = \frac{1}{b - a}, \quad a \le x \le b
$$

Used in: random initialization, Monte Carlo sampling.

```python
from scipy.stats import norm, expon, uniform

# Probability within 1 std of mean
p = norm.cdf(1) - norm.cdf(-1)
print(f"P(-1 < Z < 1) = {p:.4f}")  # 0.6827

# Exponential: P(X < 1) with rate 2
print(f"P(X < 1) = {expon.cdf(1, scale=0.5):.4f}")

# Uniform: P(0.3 < X < 0.7)
print(f"P(0.3 < X < 0.7) = {uniform.cdf(0.7) - uniform.cdf(0.3):.4f}")
```

---

## 4. Mode, Median, and Mean

| Measure | Definition | Robustness |
|:---|:---|:---|
| **Mode** | $\arg\max_x p(x)$ or $f(x)$ | Most robust to outliers |
| **Median** | $x$ where $F(x) = 0.5$ | Robust |
| **Mean** | $E[X]$ | Sensitive to outliers |

**ML Connection:** For skewed distributions (like income data), the mean is misleading. The median is often a better measure of central tendency. MAE (L1 loss) optimizes for the median; MSE (L2 loss) optimizes for the mean.

---

## 5. Entropy

The **Shannon entropy** of a distribution measures its uncertainty:

$$
H(X) = -\sum_x p(x) \log p(x) \quad \text{(discrete)}
$$

$$
H(X) = -\int f(x) \log f(x) \, dx \quad \text{(continuous, differential entropy)}
$$

**ML Connection:** Decision trees split on features that maximize **information gain** (reduction in entropy). Cross-entropy loss measures the difference between predicted and true distributions.

---

## 6. Summary

| Concept | Discrete | Continuous |
|:---|:---|:---|
| **Function** | PMF: $p(x)$ | PDF: $f(x)$ |
| **Probability** | $P(X = x) = p(x)$ | $P(X = x) = 0$ |
| **Interval** | $P(a \le X \le b) = \sum_{a}^{b} p(x)$ | $P(a \le X \le b) = \int_a^b f(x) dx$ |
| **Normalization** | $\sum p(x) = 1$ | $\int f(x) dx = 1$ |

> **Check your intuition:** Can a PDF value $f(x) = 2.5$? *(Answer: Yes! A density of 2.5 means the probability is concentrated around that point. It does NOT mean $P(X = x) = 2.5$ — for continuous variables, point probabilities are always 0.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 50: Expectation and Variance](Lecture%2050%20Expectation%20and%20Variance.md) — Statistical properties of distributions
- **Next:** [Lecture 52: Continuous Probability Distributions](Lecture%2052%20Continuous%20Probability%20Distributions.md) — Specific continuous distributions
- **Related:** [Lecture 49: Bayes Theorem and Random Variables](Lecture%2049%20Bayes%20Theorem%20and%20Random%20Variables.md) — Introduces PMF and PDF concepts
- **Related:** [Lecture 50: Expectation and Variance](Lecture%2050%20Expectation%20and%20Variance.md) — Uses PMF/PDF for calculations
- **Related:** [Lecture 53: Joint Probability Distributions](Lecture%2053%20Joint%20Probability%20Distributions.md) — Extends to multiple variables
