# Optimization Foundations & Relationship Metrics 📊

## 1. Cost Function: Mean Squared Error (MSE)

The **Cost Function** ($J$) is the mathematical objective function that quantifies the error between the model's predictions and the true target values. Minimizing this function is the primary goal of the training process.

### Formula
**Mean Squared Error (MSE)** calculates the average of the squared residuals across $n$ data points.

$$
J(\theta) = \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2
$$

| Variable | Symbol | Interpretation |
| :--- | :--- | :--- |
| **Cost Function** | $J$ | The metric to minimize. |
| **Sample Size** | $n$ | Total number of observations in the dataset. |
| **True Value** | $y_i$ | The actual ground-truth label for the $i$-th example. |
| **Prediction** | $\hat{y}_i$ | The model's output ($mx_i + b$). |
| **Squared Error** | $(\dots)^2$ | Penalizes large errors disproportionately and ensures positivity. |

---

## 2. Analytical Solution: Ordinary Least Squares (OLS)

The **Least Squares Line** ($\hat{y} = \beta_0 + \beta_1 x$) is the closed-form statistical solution that mathematically minimizes the MSE without iteration.

### Coefficient Formulas
These formulas allow direct calculation of the optimal parameters using data statistics (means and covariance).

| Parameter | Formula | Interpretation |
| :--- | :--- | :--- |
| **Slope** ($\beta_1$) | $$\beta_1 = \frac{ \sum (x_i - \bar{x})(y_i - \bar{y}) }{ \sum (x_i - \bar{x})^2 }$$ | **Rate of Change.** The estimated change in $y$ for a 1-unit increase in $x$. |
| **Intercept** ($\beta_0$) | $$\beta_0 = \bar{y} - \beta_1 \bar{x}$$ | **Bias.** The baseline value of $y$ when $x=0$. |

---

## 3. Iterative Optimization: Gradient Descent

When a direct analytical solution is computationally expensive (e.g., in high-dimensional space or neural networks), we use **Gradient Descent**.

### The Update Rule
We iteratively adjust a parameter ($\theta$) by moving in the direction opposite to the gradient (slope) of the cost function.

$$\theta_{\text{new}} = \theta_{\text{old}} - \alpha \cdot \frac{\partial J}{\partial \theta}$$

| Variable | Symbol | Role |
| :--- | :--- | :--- |
| **Parameter** | $\theta$ | The weight or bias being optimized (e.g., $\beta_0, \beta_1$). |
| **Learning Rate** | $\alpha$ | Hyperparameter controlling step size. (Too high = diverge; Too low = slow). |
| **Gradient** | $\nabla J$ | The vector of partial derivatives indicating the direction of steepest ascent. |

---

## 4. Relationship Metric: Correlation Coefficient ($r$)



The **Pearson Correlation Coefficient** ($r$) quantifies the **strength and direction** of the linear association between two variables. Unlike regression, it does not imply causation or prediction.

$$
r = \frac{ \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}) }{ \sqrt{ \sum_{i=1}^{n} (x_i - \bar{x})^2 } \cdot \sqrt{ \sum_{i=1}^{n} (y_i - \bar{y})^2 } }
$$

### Interpretation Table

| Value of $r$ | Relationship Type | Visual Intuition |
| :--- | :--- | :--- |
| **$+1.0$** | Perfect Positive Linear | Points lie exactly on an upward line. |
| **$-1.0$** | Perfect Negative Linear | Points lie exactly on a downward line. |
| **$0.0$** | No Linear Correlation | Points form a random cloud or a non-linear shape (e.g., circle). |
| **$0.7$ to $0.9$** | Strong Correlation | Clear trend with some noise. |
