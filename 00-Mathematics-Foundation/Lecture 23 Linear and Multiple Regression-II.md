## Lecture 23: Python Implementation of Regression Models

*Simple, Multiple, and Polynomial Regression — Hands-on with Scikit-Learn*

---

### The Hacker's Catch
In Lectures 21 and 22, we derived the math: normal equations, pseudo-inverses, and matrix gradients. In practice, you rarely solve these by hand. You use a **Regression Engine** (like `Scikit-Learn`). This lecture shows how to feed raw data into that engine and, more importantly, how to interpret the "black box" outputs.

---

### 1. Simple Linear Regression: Study Hours vs. Scores
The simplest case: predicting a student's score based on one feature (hours studied).

**The Model:** $Y = \beta_0 + \beta_1 X + \epsilon$

#### Core Implementation
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate Synthetic Data (replace with pd.read_csv('student_scores.csv') for real data)
np.random.seed(42)
X = np.random.uniform(1, 10, 50).reshape(-1, 1)  # Study hours
y = 9.5 * X.flatten() + 5 + np.random.randn(50) * 3  # Scores with noise

# 2. Build the Model
reg = LinearRegression()
reg.fit(X, y)

# 3. Extract the Math
print(f"Intercept (Beta 0): {reg.intercept_:.2f}")
print(f"Slope (Beta 1): {reg.coef_[0]:.2f}")
print(f"R² Score: {reg.score(X, y):.4f}")
```

#### The Intuition
The `reg.coef_` tells you exactly how much your "Score" jumps for every 1-hour increase in "Study Time". If `coef_` is 9.74, then for every hour you grind, you gain ~9.7 points. The `intercept_` is your "baseline" score if you studied zero hours.

---

### 2. Multiple Linear Regression: The Startup Profit Engine
Real-world data is multivariate. Here, we predict `Profit` based on `R&D Spend`, `Administration`, and `Marketing`.

**The Model:** $Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \epsilon$

#### Implementation Workflow
```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Generate Synthetic Startup Data
np.random.seed(42)
n = 500
rd_spend = np.random.uniform(10000, 200000, n)
admin_spend = np.random.uniform(50000, 150000, n)
marketing = np.random.uniform(50000, 300000, n)
profit = 0.6 * rd_spend + 0.1 * admin_spend + 0.05 * marketing + 5000 + np.random.randn(n) * 10000

X = np.column_stack([rd_spend, admin_spend, marketing])
y = profit

# 2. Train/Test Split (The Gold Standard)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 3. Fit on Training Data
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 4. Predict on UNSEEN Data
y_pred = regressor.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"Coefficients: {regressor.coef_}")
```

> **Pro-Tip:** Always evaluate your model on `X_test`. If you only evaluate on the data the model has already seen (`X_train`), you are testing the model's **memory**, not its **intelligence**.

---

### 3. Polynomial Regression: The "Bulb Weight" Curve
Sometimes a straight line is too stupid. If data curves (like the growth of an onion bulb over time), we need higher-degree terms.

**The Model (2nd Degree):** $Y = \beta_0 + \beta_1 X + \beta_2 X^2$

#### Implementation (Using NumPy's Poly Utility)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Generate curved data (bulb weight over time)
np.random.seed(42)
X = np.linspace(0, 30, 50).reshape(-1, 1)
y = 2.0 * X.flatten()**1.5 + 10 + np.random.randn(50) * 5

# Fit polynomials of different degrees
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, degree in zip(axes, [1, 3, 12]):
    coeffs = np.polyfit(X.flatten(), y, degree)
    model = np.poly1d(coeffs)
    y_pred = model(X.flatten())
    ax.scatter(X, y, s=20, alpha=0.6)
    ax.plot(X, y_pred, 'r-', lw=2)
    ax.set_title(f"Degree {degree} (R²={r2_score(y, y_pred):.3f})")
plt.tight_layout()
plt.show()
```

---

### 4. The Overfitting Trap
How do we know which degree to choose?
- **Degree 1 (Underfitting):** Too simple. High Error. Doesn't capture the trend.
- **Degree 5 (Just Right):** Captures the curve. Generalizes well to new points.
- **Degree 12 (Overfitting):** Looks perfect on training data (Error $\approx 0$). It literally "connects the dots." But it wiggles wildly between points!

**The Result:** A 12th-degree model might predict a negative weight for a bulb at 14.5 days just because it's trying to fit the noise. This is why we check **R² Score** and **Mean Squared Error (MSE)** on test data.

---

### Evaluation Metrics

| Metric | Formula | Goal | Intuition |
| :--- | :--- | :--- | :--- |
| **MSE** | $\frac{1}{n} \Sigma (y - \hat{y})^2$ | **Minimize** | Penalizes large errors heavily (due to squaring). |
| **R² Score** | $1 - \frac{SS_{res}}{SS_{total}}$ | **Get to 1.0** | Tells you what % of the data's "story" your model understands. |

---

### Practical Application
- **Supervised Learning Lab:** [multivariate_regression_lab.py](../02-Supervised-Learning/LINEAR%20REGRESSION%20WITH%20MULTIPLE%20VARIABLES/multivariate_regression_lab.py)
- **Mathematical Link:** Re-read [Lecture 21 (Least Squares)](Lecture%2021%20Least%20Square%20Approximation%20and%20Minimum%20Norm%20Solution.md) to see how the computer actually solves for these coefficients using the Pseudo-Inverse.
- **Core Concept Link:** Deep dive into the [Bias-Variance Tradeoff](../../01-Core-Concepts/Bias-Variance-Tradeoff.md) to understand why high-degree polynomials (Overfitting) are dangerous.

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 22: Linear and Multiple Regression](Lecture%2022%20Linear%20and%20Multiple%20Regression.md) — Mathematical foundation of linear regression models
- **Next:** [Lecture 24: Logistic Regression-I](Lecture%2024%20Logistic%20Regression-I.md) — Classification via logistic (sigmoid) transformation
- **Related:** [Lecture 25: Logistic Regression-II](Lecture%2025%20Logistic%20Regression-II.md) — Python implementation of logistic regression
