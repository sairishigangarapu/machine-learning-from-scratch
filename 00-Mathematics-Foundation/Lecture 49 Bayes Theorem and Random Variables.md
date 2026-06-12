## Bayes' Theorem and Random Variables

*Essential Mathematics for ML — Structured Notes*

---

## 1. Bayes' Theorem

### Motivation and Intuition
You test positive for a disease that affects 1 in 10,000 people. The test is 99% accurate. Should you panic? Probably not — the prior probability of having the disease is so low that even a positive test is more likely to be a false positive than a true positive. **Bayes' Theorem** is the mathematical engine that flips conditional probabilities, and it is the theoretical foundation of Bayesian inference, spam filtering, medical diagnosis, and more.

### The Formula

$$
P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}
$$

| Term | Name | Meaning |
|:---|:---|:---|
| $P(A)$ | **Prior** | Initial belief about $A$ before seeing evidence |
| $P(B | A)$ | **Likelihood** | How likely is the evidence if $A$ is true? |
| $P(B)$ | **Evidence** | Overall probability of the evidence |
| $P(A | B)$ | **Posterior** | Updated belief about $A$ after seeing evidence |

### Derivation

From the definition of conditional probability:

$$
P(A | B) = \frac{P(A \cap B)}{P(B)}, \quad P(B | A) = \frac{P(A \cap B)}{P(A)}
$$

Solving: $P(A \cap B) = P(A | B) \cdot P(B) = P(B | A) \cdot P(A)$

$$
\therefore \quad P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}
$$

```python
def bayes_theorem(p_a, p_b_given_a, p_b):
    """Compute P(A|B) using Bayes' theorem."""
    return (p_b_given_a * p_a) / p_b

# Disease example
p_disease = 1 / 10000      # Prior: 1 in 10,000
p_pos_given_disease = 0.99  # Sensitivity: 99%
p_pos_given_healthy = 0.01  # False positive rate: 1%

# Total probability of positive test
p_positive = (p_pos_given_disease * p_disease + 
              p_pos_given_healthy * (1 - p_disease))

# Posterior
p_disease_given_pos = bayes_theorem(p_disease, p_pos_given_disease, p_positive)
print(f"P(disease | positive) = {p_disease_given_pos:.4f}")  # ~0.0098
# Despite 99% accuracy, only ~1% chance of actually having the disease!
```

---

## 2. Random Variables

### Definition
A **random variable** $X$ is a function that maps outcomes from the sample space to real numbers:

$$
X: \Omega \to \mathbb{R}
$$

### Types

| Type | Description | Example |
|:---|:---|:---|
| **Discrete** | Takes countable values | Number of spam emails (0, 1, 2, ...) |
| **Continuous** | Takes any value in an interval | Height, temperature, stock price |

### Probability Mass Function (PMF) — Discrete

$$
p(x) = P(X = x)
$$

Properties: $p(x) \ge 0$ and $\sum_x p(x) = 1$.

### Probability Density Function (PDF) — Continuous

$$
P(a \le X \le b) = \int_a^b f(x) \, dx
$$

Properties: $f(x) \ge 0$ and $\int_{-\infty}^{\infty} f(x) \, dx = 1$.

Note: For continuous variables, $P(X = x) = 0$ for any specific point — probability is defined over intervals.

```python
import numpy as np

# Discrete: flip a fair coin, X = number of heads in 3 flips
# PMF: P(X=0)=1/8, P(X=1)=3/8, P(X=2)=3/8, P(X=3)=1/8
from scipy.stats import binom

x = np.arange(0, 4)
pmf = binom.pmf(x, n=3, p=0.5)
print(f"PMF: {dict(zip(x, pmf.round(4)))}")
```

---

## 3. Cumulative Distribution Function (CDF)

$$
F(x) = P(X \le x)
$$

Properties:
* Non-decreasing
* $\lim_{x \to -\infty} F(x) = 0$, $\lim_{x \to \infty} F(x) = 1$
* For discrete: $F(x) = \sum_{t \le x} p(t)$
* For continuous: $F(x) = \int_{-\infty}^x f(t) \, dt$

```python
from scipy.stats import norm

# CDF of standard normal at x = 0, 1, 2
print(f"F(0) = {norm.cdf(0):.4f}")   # 0.5000
print(f"F(1) = {norm.cdf(1):.4f}")   # 0.8413
print(f"F(2) = {norm.cdf(2):.4f}")   # 0.9772
```

---

## 4. Common Discrete Distributions

| Distribution | PMF | Parameters | ML Use Case |
|:---|:---|:---|:---|
| **Bernoulli** | $p^x(1-p)^{1-x}$ | $p$ | Binary classification |
| **Binomial** | $\binom{n}{x}p^x(1-p)^{n-x}$ | $n, p$ | Number of successes |
| **Poisson** | $\frac{\lambda^x e^{-\lambda}}{x!}$ | $\lambda$ | Event counts (rare events) |
| **Categorical** | $p_k$ for class $k$ | $p_1, \dots, p_K$ | Multi-class labels |

```python
from scipy.stats import poisson

# P(X = 3) when lambda = 2 (average 2 events per interval)
print(f"P(X=3) = {poisson.pmf(3, mu=2):.4f}")  # 0.1804

# P(X <= 2)
print(f"P(X<=2) = {poisson.cdf(2, mu=2):.4f}")  # 0.6767
```

---

## 5. Common Continuous Distributions

| Distribution | PDF | Parameters | ML Use Case |
|:---|:---|:---|:---|
| **Uniform** | $\frac{1}{b-a}$ | $a, b$ | Random initialization |
| **Normal (Gaussian)** | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu, \sigma$ | Feature modeling, VAEs |
| **Exponential** | $\lambda e^{-\lambda x}$ | $\lambda$ | Waiting times, survival analysis |

### The Gaussian Distribution

The most important distribution in ML. The **Central Limit Theorem** states that the sum of many independent random variables tends toward a Gaussian, which is why it appears everywhere.

**Standard Normal:** $\mu = 0, \sigma = 1$

$$
\phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}
$$

```python
import numpy as np
from scipy.stats import norm

# 68-95-99.7 rule
mu, sigma = 0, 1
print(f"P(|X-mu| < 1*sigma) = {norm.cdf(1) - norm.cdf(-1):.4f}")  # 0.6827
print(f"P(|X-mu| < 2*sigma) = {norm.cdf(2) - norm.cdf(-2):.4f}")  # 0.9545
print(f"P(|X-mu| < 3*sigma) = {norm.cdf(3) - norm.cdf(-3):.4f}")  # 0.9973
```

---

## 6. Bayes' Theorem in ML: Naive Bayes

The Naive Bayes classifier applies Bayes' theorem with the **naive** assumption that features are conditionally independent:

$$
P(y | x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i | y)
$$

For text classification (spam detection):

$$
P(\text{spam} | \text{words}) \propto P(\text{spam}) \prod_{i} P(\text{word}_i | \text{spam})
$$

Each $P(\text{word}_i | \text{spam})$ is estimated from training data.

---

## 7. Summary

| Concept | Formula | ML Role |
|:---|:---|:---|
| **Bayes' Theorem** | $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$ | Flip conditioning, Bayesian inference |
| **PMF** | $P(X=x)$ | Discrete distributions |
| **PDF** | $f(x)$, $\int f = 1$ | Continuous distributions |
| **CDF** | $F(x) = P(X \le x)$ | Cumulative probabilities |
| **Gaussian** | $\mathcal{N}(\mu, \sigma^2)$ | Ubiquitous in ML |

> **Check your intuition:** If $X \sim \mathcal{N}(0, 1)$, what is $P(-1 < X < 1)$? *(Answer: About 68%. The 68-95-99.7 rule says one standard deviation from the mean captures ~68% of the probability.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 48: Introduction to Probability](Lecture%2048%20Introduction%20to%20Probability.md) — Basic probability axioms and rules
- **Next:** [Lecture 50: Expectation and Variance](Lecture%2050%20Expectation%20and%20Variance.md) — Quantifies properties of distributions
- **Related:** [Lecture 48: Introduction to Probability](Lecture%2048%20Introduction%20to%20Probability.md) — Foundation for Bayes' theorem
- **Related:** [Lecture 51: PMF and PDF](Lecture%2051%20PMF%20and%20PDF.md) — Mathematical representation of random variables
