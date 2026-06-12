## Joint Probability Distributions

*Essential Mathematics for ML — Structured Notes*

---

## 1. Joint Distribution

### Motivation and Intuition
Real-world data has multiple features that are often correlated. A patient's blood pressure and cholesterol level are not independent — knowing one tells you something about the other. The **joint distribution** captures the probability of specific combinations of values for multiple random variables.

### Definition
For discrete random variables $X$ and $Y$:

$$
p(x, y) = P(X = x, Y = y)
$$

**Properties:**
* $p(x, y) \ge 0$
* $\sum_x \sum_y p(x, y) = 1$

### Joint PDF (Continuous)

$$
P(a \le X \le b, \; c \le Y \le d) = \int_c^d \int_a^b f(x, y) \, dx \, dy
$$

```python
import numpy as np

# Joint PMF of two dice
p = np.ones((6, 6)) / 36

# P(X=3, Y=4)
print(f"P(X=3, Y=4) = {p[2, 3]:.4f}")  # 0.0278

# P(sum = 7)
sum7 = sum(p[i, 6-i-1] for i in range(6))
print(f"P(X+Y=7) = {sum7:.4f}")  # 0.1667
```

---

## 2. Marginal Distribution

The **marginal distribution** of $X$ is obtained by summing (or integrating) out $Y$:

### Discrete
$$
p_X(x) = \sum_y p(x, y)
$$

### Continuous
$$
f_X(x) = \int_{-\infty}^{\infty} f(x, y) \, dy
$$

**Intuition:** The marginal distribution answers: "What is the probability of $X$ regardless of what $Y$ is?"

```python
import numpy as np

# Joint distribution of height and weight
joint = np.array([[0.1, 0.05, 0.02],
                  [0.05, 0.2, 0.1],
                  [0.02, 0.1, 0.36]])

# Marginal of X (sum over columns)
marginal_x = joint.sum(axis=1)
print(f"Marginal X: {marginal_x}")  # [0.17, 0.35, 0.48]

# Marginal of Y (sum over rows)
marginal_y = joint.sum(axis=0)
print(f"Marginal Y: {marginal_y}")  # [0.17, 0.35, 0.48]
```

---

## 3. Conditional Distribution

$$
p(y | x) = \frac{p(x, y)}{p_X(x)}, \quad p_X(x) > 0
$$

**Intuition:** Given that we observed $X = x$, what is the probability distribution of $Y$?

```python
import numpy as np

joint = np.array([[0.1, 0.05, 0.02],
                  [0.05, 0.2, 0.1],
                  [0.02, 0.1, 0.36]])

# Conditional distribution of Y given X=1
x_given = 1
marginal_x = joint.sum(axis=1)
conditional_y = joint[x_given, :] / marginal_x[x_given]
print(f"P(Y | X=1): {conditional_y}")  # [0.143, 0.571, 0.286]
```

---

## 4. Independence

$X$ and $Y$ are **independent** if:

$$
p(x, y) = p_X(x) \cdot p_Y(y) \quad \forall x, y
$$

Equivalently: $p(y | x) = p_Y(y)$ — knowing $X$ tells you nothing about $Y$.

**ML Connection:** The Naive Bayes classifier assumes conditional independence of features given the class label. This is almost always violated in practice, but the classifier still works.

---

## 5. Covariance and Correlation

### Covariance
$$
\text{Cov}(X, Y) = E[XY] - E[X]E[Y]
$$

### Correlation
$$
\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]
$$

| $\rho$ | Meaning |
|:---|:---|
| $+1$ | Perfect positive linear relationship |
| $0$ | No linear relationship |
| $-1$ | Perfect negative linear relationship |

**Important:** $\rho = 0$ does NOT mean independence. It only means no *linear* relationship. Two variables can be perfectly dependent but uncorrelated (e.g., $Y = X^2$ with $X \sim \mathcal{U}(-1, 1)$).

```python
import numpy as np

# Uncorrelated but dependent
X = np.random.uniform(-1, 1, 10000)
Y = X**2

corr = np.corrcoef(X, Y)[0, 1]
print(f"Corr(X, X^2) = {corr:.4f}")  # ~0 (uncorrelated)
# But Y is completely determined by X!
```

---

## 6. The Covariance Matrix

For a random vector $\mathbf{X} = (X_1, \dots, X_n)^T$:

$$
\Sigma_{ij} = \text{Cov}(X_i, X_j)
$$

Properties:
* Symmetric: $\Sigma = \Sigma^T$
* Positive semi-definite: $\mathbf{a}^T \Sigma \mathbf{a} \ge 0$
* Diagonal entries are variances, off-diagonals are covariances

**ML Connection:** PCA (Lecture 16) eigendecomposes the covariance matrix to find directions of maximum variance. The covariance matrix is the sufficient statistic for Gaussian distributions.

---

## 7. Summary

| Concept | Formula | Question Answered |
|:---|:---|:---|
| **Joint** | $p(x, y)$ | "What is $P(X=x \text{ and } Y=y)$?" |
| **Marginal** | $p_X(x) = \sum_y p(x,y)$ | "What is $P(X=x)$ regardless of $Y$?" |
| **Conditional** | $p(y|x) = p(x,y)/p_X(x)$ | "Given $X=x$, what is $P(Y=y)$?" |
| **Independence** | $p(x,y) = p_X(x)p_Y(y)$ | "Does knowing $X$ tell me about $Y$?" |

> **Check your intuition:** If $X$ and $Y$ are independent, is $\text{Cov}(X, Y) = 0$? *(Answer: Yes. Independence implies zero covariance. But zero covariance does NOT imply independence — only zero correlation for linear relationships.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 52: Continuous Probability Distributions](Lecture%2052%20Continuous%20Probability%20Distributions.md) — Continuous single-variable distributions
- **Next:** [Lecture 54: Introduction to Support Vector Machines](Lecture%2054%20Introduction%20to%20Support%20Vector%20Machines.md) — Applies probability to classification
- **Related:** [Lecture 50: Expectation and Variance](Lecture%2050%20Expectation%20and%20Variance.md) — Covariance and correlation concepts
- **Related:** [Lecture 51: PMF and PDF](Lecture%2051%20PMF%20and%20PDF.md) — Distribution functions for multiple variables
- **Related:** [Lecture 52: Continuous Probability Distributions](Lecture%2052%20Continuous%20Probability%20Distributions.md) — Continuous multivariate distributions
- **Related:** [Lecture 16: Principal Component Analysis](Lecture%2016%20Principal%20Component%20Analysis.md) — Uses covariance matrix for dimensionality reduction
