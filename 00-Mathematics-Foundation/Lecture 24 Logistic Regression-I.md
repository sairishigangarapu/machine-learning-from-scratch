## Lecture 24: Logistic Regression - I

### 1. Introduction to Logistic Regression
Logistic regression is the most popular algorithm for classification problems. Unlike linear regression, where the dependent variable is continuous, logistic regression handles discrete output variables. In this lecture, we will focus on classification—the process of assigning a data point to a specific category.

#### Binary vs. Multinomial Classification
1. **Binary Classification:** The output variable has only two possible values.
   - Examples: Weather prediction (Rain: Yes/No), Medical diagnosis (Covid: Positive/Negative), Logic (True/False or 0/1).
2. **Multinomial Classification:** The output variable can take more than two discrete values.
   - Example: Categorizing a student's performance (Very Good, Good, Average, Poor).

---

### 2. Why not Linear Regression for Classification?
One might question why we cannot simply use the Least Square method (Linear Regression) for classification. In Linear Regression, we find parameters $\beta$ that minimize the residual error:

$$ \min_{\beta} \sum_{i=1}^{n} (y_i - \beta^T x_i)^2 $$

This produces a model $y = \beta^T x$. For any new input $x$, the model returns a value $y$ from the set of real numbers $(-\infty, \infty)$. However, in binary classification, we need $y$ to be strictly 0 or 1. Mapping a continuous real line to discrete classes is problematic. Instead of predicting the class directly, we must think in terms of **Probability**.

---

### 3. The Sigmoid Function
To map the infinite real line to the probability interval $[0, 1]$, we utilize a **Sigmoid Function**. A sigmoid function has a characteristic S-shaped curve. Common examples include:
- **Logistic Function:** $s(z) = \frac{1}{1 + e^{-z}}$ (Used in Logistic Regression)
- **Hyperbolic Tangent:** $\tanh(z)$
- **Arc Tangent:** $\arctan(z)$

#### Properties of the Logistic Function
The function $s(z) = \frac{1}{1 + e^{-z}}$ has ideal properties for classification:
1. **Bounded:** As $z \to \infty$, $s(z) \to 1$. As $z \to -\infty$, $s(z) \to 0$.
2. **Probability Mapping:** Every output is strictly between 0 and 1.
3. **Smoothness:** It is infinitely differentiable, which is critical for optimization algorithms like Gradient Descent.

---

### 4. Probability Foundations
Before formalizing the model, we review the basics:
- **Sample Space ($\Omega$):** The set of all possible outcomes.
  - Tossing 2 coins: $\{HH, HT, TH, TT\}$.
  - Rolling a die: $\{1, 2, 3, 4, 5, 6\}$.
- **Event Space:** A collection of subsets of the sample space.
- **Probability ($P$):** A number associated with an event such that $0 \le P \le 1$ and the sum of probabilities of all possible events is 1.
- **Random Variable ($X$):** A function from the sample space to a target space (the classes).
  - Example: Counting the number of heads in two coin tosses maps $\{TT\} \to 0$, $\{HT, TH\} \to 1$, and $\{HH\} \to 2$.

---

### 5. The Logistic Regression Model
Instead of modeling $y$ directly as a linear combination, we model the probability that $y=1$ for a given $x$:

$$ p(x) = P(y=1|x) = \frac{e^{\alpha^T x}}{1 + e^{\alpha^T x}} = \sigma(\alpha^T x) $$

#### Deriving the Logit (Log-Odds)
If $p(x)$ is the probability of class 1, then the probability of class 0 is:
$$ 1 - p(x) = 1 - \frac{e^{\alpha^T x}}{1 + e^{\alpha^T x}} = \frac{1}{1 + e^{\alpha^T x}} $$

Taking the ratio (the Odds Ratio):
$$ \frac{p(x)}{1 - p(x)} = \frac{e^{ \alpha^T x} / (1 + e^{\alpha^T x})}{1 / (1 + e^{\alpha^T x})} = e^{\alpha^T x} $$

Taking the natural logarithm:
$$ \log\left(\frac{p(x)}{1 - p(x)}\right) = \alpha^T x = \alpha_0 + \alpha_1 x_1 + \dots + \alpha_p x_p $$

The left-hand side is called the **Logit** (or **Logos**) of $p(x)$. It shows that the log-odds of the probability are a linear combination of the features.

---

### 6. Maximum Likelihood Estimation (MLE)
To find the optimal coefficients $\alpha$, we use Maximum Likelihood Estimation. Given $n$ data points $(x_i, y_i)$, the Likelihood Function $L(\alpha)$ represents the probability of observing our specific data:

$$ L(\alpha) = \prod_{i=1}^{n} p(x_i)^{y_i} (1 - p(x_i))^{1 - y_i} $$

#### Final Log-Likelihood Form
To maximize $L(\alpha)$, we maximize its logarithm $l(\alpha)$:
$$ l(\alpha) = \sum_{i=1}^{n} \left[ y_i \log p(x_i) + (1 - y_i) \log(1 - p(x_i)) \right] $$

Substituting $p(x_i)$ and simplifying:
$$ l(\alpha) = \sum_{i=1}^{n} \left[ y_i (\alpha^T x_i) - \log(1 + e^{\alpha^T x_i}) \right] $$

#### The Optimization Wall
Setting the derivative $\frac{\partial l}{\partial \alpha} = 0$ does not result in a closed-form solution (like the Pseudo-inverse in OLS). We must use numerical optimization techniques, such as **Gradient Descent**, to find the optimal $\alpha$.

---

### 7. Making Predictions
For an unknown pattern $x^*$:
1. Calculate the linear score: $m = \alpha^T x^*$.
2. Pass it through the sigmoid: $y = \frac{1}{1 + e^{-m}}$.
3. **Thresholding:** 
   - If $y \ge 0.5 \implies$ Class 1
   - If $y < 0.5 \implies$ Class 0

---

### 8. Multiclass Extension
For problems with more than 2 classes:
1. **One-Hot Encoding:** Convert target classes into vector labels.
2. **Softmax Function:** Replace the logistic sigmoid with the Softmax function to handle vector-valued probability outputs.

---

### Practical Application
- **Supervised Classification Lab:** See the 02-Supervised-Learning module for the Logistic Regression implementation.
- **Comparison:** Compare these derivations with [Lecture 22 (Linear Regression)](Lecture%2022%20Linear%20and%20Multiple%20Regression.md) to see the transition from Least Squares to Maximum Likelihood.

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 23: Linear and Multiple Regression-II](Lecture%2023%20Linear%20and%20Multiple%20Regression-II.md) — Python implementation of linear regression
- **Next:** [Lecture 25: Logistic Regression-II](Lecture%2025%20Logistic%20Regression-II.md) — Python implementation of logistic regression with gradient descent
- **Related:** [Lecture 22: Linear and Multiple Regression](Lecture%2022%20Linear%20and%20Multiple%20Regression.md) — Linear regression as the regression counterpart to logistic classification
- **Related:** [Lecture 48: Introduction to Probability](Lecture%2048%20Introduction%20to%20Probability.md) — Probability foundations for MLE derivation
