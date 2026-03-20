## Lecture 24: Logistic Regression - I

*The Mathematical Bridge to Classification — Sigmoids, Odds, and MLE*

---

### The Hacker's Catch
Standard Linear Regression predicts a continuous number (like house price). But what if you want to predict a **Category** (Spam vs. Not Spam, 0 vs. 1)? You can't just use a straight line because a line goes to $+\infty$ and $-\infty$, but a probability must stay between $0$ and $1$. 

**Logistic Regression** is the solution: it takes a linear engine and "squashes" the output through a mathematical funnel called the **Sigmoid Function**.

---

### 1. The Probability Squasher: The Sigmoid Function
To map any real-valued number $z$ (from our linear equation $z = \alpha^T x$) to a probability $P \in [0, 1]$, we use the **Logistic (Sigmoid) Function**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}
$$

#### Geometric Intuition
- **$z \to \infty$:** $e^{-z} \to 0$, so $\sigma(z) \to 1$.
- **$z \to -\infty$:** $e^{-z}$ explodes, so $\sigma(z) \to 0$.
- **$z = 0$:** $\sigma(0) = 0.5$ (The perfectly undecided state).

The curve is smooth (infinitely differentiable), which is a "Gift from God" for optimization algorithms like Gradient Descent.

---

### 2. The Logit Link (Log-Odds)
In Linear Regression, we modeled $y = \alpha^T x$. In Logistic Regression, we model the **Log-Odds** (Logit):

$$
\log\left(\frac{p(x)}{1 - p(x)}\right) = \alpha_0 + \alpha_1 x_1 + \dots + \alpha_p x_p = \alpha^T x
$$

#### Why the Logit?
1. **$p(x)$** is in $[0, 1]$.
2. **$\frac{p(x)}{1 - p(x)}$** (The Odds) is in $[0, \infty)$.
3. **$\log(\text{Odds})$** is in $(-\infty, \infty)$.

By taking the log, we transform the restricted probability space into an unrestricted realm that can be modeled by a standard linear combination of features.

---

### 3. Maximum Likelihood Estimation (MLE)
Unlike Linear Regression (where we use the "Least Squares" closed-form solution), Logistic Regression uses **Maximum Likelihood Estimation**. We don't have a "Normal Equation" pseudo-inverse here because the sigmoid function is non-linear.

#### The Likelihood Function
We seek parameters $\alpha$ that maximize the probability of observing our actual training data $(x_i, y_i)$:

$$
L(\alpha) = \prod_{i:y_i=1} p(x_i) \prod_{i:y_i=0} (1 - p(x_i))
$$

#### The Log-Likelihood ($l(\alpha)$)
To make the math tractable, we maximize the **Log** of the likelihood (since log is monotonically increasing):

$$
l(\alpha) = \sum_{i=1}^{n} \left[ y_i \log p(x_i) + (1 - y_i) \log(1 - p(x_i)) \right]
$$

> **The Optimization Wall:** Setting the derivative of $l(\alpha)$ to zero does **not** yield a simple matrix solution. We must solve this iteratively using **Gradient Descent** or Newton's Method.

---

### 4. Making the Prediction (The Decision Boundary)
Once the coefficients $\alpha$ are optimized:
1.  Calculate $z = \alpha^T x_{new}$.
2.  Calculate $\hat{y} = \sigma(z)$.
3.  **Apply Threshold:** By default, if $\hat{y} \ge 0.5 \implies$ Class 1. Otherwise $\implies$ Class 0.

---

### Practical Application
- **Supervised Classification Lab:** Explore the implementation in [Logistic Regression Theory Lab](../../02-Supervised-Learning/LOGISTIC%20REGRESSION/Theory.md).
- **Mathematical Link:** Compare this to [Lecture 22 (Linear Regression)](../../00-Mathematics-Foundation/Lecture%2022%20Linear%20and%20Multiple%20Regression.md) to understand why the sigmoid is necessary.
- **Multiclass Extension:** For more than 2 classes, notice the jump from Sigmoid to **Softmax** and **One-Hot Encoding**.
