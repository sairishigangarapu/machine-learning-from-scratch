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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $P(A \mid B)$ | **Posterior** — updated probability of $A$ given evidence $B$ | The output of Bayes' theorem; combines prior knowledge with observed data to produce an updated belief |
| $P(B \mid A)$ | **Likelihood** — probability of evidence $B$ assuming $A$ is true | Connects observed data to the hypothesis; estimated from training data in ML |
| $P(A)$ | **Prior** — initial belief about $A$ before seeing evidence | Encodes domain knowledge or historical base rates; crucial in medical diagnosis and spam filtering |
| $P(B)$ | **Evidence** — total probability of $B$ across all hypotheses | Normalizes the posterior; computed via the law of total probability: $\sum P(B \mid A)P(A)$ |

### Derivation

From the definition of conditional probability:

$$
P(A | B) = \frac{P(A \cap B)}{P(B)}, \quad P(B | A) = \frac{P(A \cap B)}{P(A)}
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $P(A \mid B)$ | Conditional probability of $A$ given $B$ | Defines the posterior in Bayesian inference — updated belief after seeing evidence |
| $P(B \mid A)$ | Conditional probability of $B$ given $A$ | The likelihood — how probable the evidence is under the assumption $A$ is true |
| $P(A \cap B)$ | Joint probability of both $A$ and $B$ | Connects both conditional formulas; is the same quantity in both numerators |
| $P(A)$ | Marginal probability of $A$ | The prior — belief before seeing evidence |
| $P(B)$ | Marginal probability of $B$ | The evidence — normalizing constant that ensures the posterior sums to 1 |

Solving: $P(A \cap B) = P(A | B) \cdot P(B) = P(B | A) \cdot P(A)$

$$
\therefore \quad P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $P(A \mid B)$ | **Posterior** probability — belief about $A$ after seeing $B$ | The output of Bayes' theorem; combines prior knowledge with observed evidence |
| $P(B \mid A)$ | **Likelihood** — probability of evidence $B$ under hypothesis $A$ | Connects the observed data to the hypothesis; estimated from training data in ML |
| $P(A)$ | **Prior** probability — initial belief about $A$ before any evidence | Encodes domain knowledge or historical base rates; crucial in medical diagnosis and spam filtering |
| $P(B)$ | **Evidence** — total probability of $B$ across all hypotheses | Normalizes the result; computed via the law of total probability: $\sum P(B \mid A) P(A)$ |

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

| Term | Definition | Significance |
|:---|:---|:---|
| $X$ | **Random variable** — a function from outcomes to real numbers | Bridges sample space (abstract outcomes) to real-valued quantities we can manipulate algebraically |
| $\Omega$ | **Sample space** — set of all possible outcomes | The domain of $X$; every outcome $\omega \in \Omega$ maps to a real number $X(\omega)$ |
| $\mathbb{R}$ | **Real numbers** — the codomain of $X$ | Random variables map to $\mathbb{R}$ so we can compute expectations, variances, and use calculus on distributions |
| $\to$ | Function mapping | Indicates $X$ assigns a numerical value to each outcome; the machinery for translating uncertainty into numbers |

### Types

| Type | Description | Example |
|:---|:---|:---|
| **Discrete** | Takes countable values | Number of spam emails (0, 1, 2, ...) |
| **Continuous** | Takes any value in an interval | Height, temperature, stock price |

### Probability Mass Function (PMF) — Discrete

$$
p(x) = P(X = x)
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $p(x)$ | **Probability Mass Function** — probability that $X$ equals exactly $x$ | Defines the distribution of a discrete random variable; every discrete distribution is characterized by its PMF |
| $P(X = x)$ | Probability that random variable $X$ takes the specific value $x$ | For discrete $X$, this can be non-zero; for continuous $X$, $P(X = x) = 0$ always |
| $X$ | Discrete random variable | Takes values from a countable set (e.g., number of heads, class label) |

Properties: $p(x) \ge 0$ and $\sum_x p(x) = 1$.

### Probability Density Function (PDF) — Continuous

$$
P(a \le X \le b) = \int_a^b f(x) \, dx
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $P(a \le X \le b)$ | **Probability** that $X$ falls in interval $[a, b]$ | For continuous variables, probability is only defined over intervals, not at single points |
| $\int_a^b f(x) \, dx$ | **Integral** of the PDF from $a$ to $b$ | Area under the PDF gives probability; the total area under $f(x)$ from $-\infty$ to $\infty$ must be $1$ |
| $f(x)$ | **Probability Density Function** — density at point $x$ | Not a probability itself ($f(x) > 1$ is possible); it is the rate at which probability accumulates |
| $a, b$ | Interval endpoints | Define the range over which we integrate to compute probability |

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

| Term | Definition | Significance |
|:---|:---|:---|
| $F(x)$ | **Cumulative Distribution Function** — probability that $X$ is at most $x$ | Uniquely defines any distribution (discrete or continuous); always exists and is well-defined |
| $P(X \le x)$ | Probability that the random variable takes a value less than or equal to $x$ | Monotonic non-decreasing function; $F(-\infty) = 0$ and $F(\infty) = 1$ |
| $X$ | Random variable (discrete or continuous) | The CDF works for both types, making it the universal representation of a distribution |

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

| Term | Definition | Significance |
|:---|:---|:---|
| $\phi(x)$ | **Standard normal PDF** — density of $Z \sim \mathcal{N}(0, 1)$ | The Gaussian in its simplest form; any normal $X \sim \mathcal{N}(\mu, \sigma^2)$ can be standardized to $Z$ |
| $\frac{1}{\sqrt{2\pi}}$ | **Normalization constant** | Ensures $\int_{-\infty}^{\infty} \phi(x) \, dx = 1$; makes $\phi$ a valid probability density |
| $e^{-x^2/2}$ | **Exponential kernel** | Gives the bell shape; decays rapidly for large $x$, making extreme values unlikely |
| $x$ | Value of the standard normal variable | Measured in standard deviations from the mean (which is $0$ for $Z$) |

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

| Term | Definition | Significance |
|:---|:---|:---|
| $P(y \mid x_1, \dots, x_n)$ | **Posterior** probability of class $y$ given all features | What the Naive Bayes classifier computes for each possible class; the argmax over $y$ is the predicted label |
| $P(y)$ | **Prior** probability of class $y$ | Captures class imbalance before seeing any features |
| $P(x_i \mid y)$ | **Likelihood** of feature $x_i$ given class $y$ | The "naive" assumption: features are conditionally independent given the class, so their likelihoods multiply |
| $\prod_{i=1}^{n}$ | Product over all $n$ features | Multiplying instead of computing a full joint distribution reduces parameters from exponential to linear in $n$ |
| $\propto$ | Proportional to | The denominator $P(x_1, \dots, x_n)$ is constant across classes and does not affect the argmax decision |

For text classification (spam detection):

$$
P(\text{spam} | \text{words}) \propto P(\text{spam}) \prod_{i} P(\text{word}_i | \text{spam})
$$

| Term | Definition | Significance |
|:---|:---|:---|
| $P(\text{spam} \mid \text{words})$ | **Posterior** — probability an email is spam given its words | The actual output used to classify each incoming email |
| $P(\text{spam})$ | **Prior** — overall spam rate in the email corpus | Encodes base-rate information; if 80% of email is spam, this term dominates |
| $P(\text{word}_i \mid \text{spam})$ | **Likelihood** of the $i$-th word in a spam email | Estimated from the frequency of each word in spam training emails |
| $\prod_{i}$ | Product over all words in the email | The naive independence assumption lets us multiply word probabilities instead of modeling word co-occurrence |

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
