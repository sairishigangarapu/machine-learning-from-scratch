# Multivariate Linear Regression

## 1. Concept Overview
**Multivariate Linear Regression** extends the simple linear model to accommodate multiple independent variables (features). Instead of fitting a **line** in 2D space, the algorithm fits a **hyperplane** in $n$-dimensional space.

### The Core Logic
In real-world scenarios, a target variable (like House Price) depends on multiple factors (Area, Bedrooms, Age), not just one. The model assigns a specific **weight** (coefficient) to each feature to determine its contribution to the final prediction.

---

## 2. Mathematical Representation

The general formula extends the simple $y=mx+c$ equation:

$$
y = m_1x_1 + m_2x_2 + m_3x_3 + \dots + m_nx_n + b
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $y$ | **Predicted target** value | The output of the multivariate regression model; a linear combination of all weighted features plus bias |
| $m_i$ | **Coefficient** (weight) for the $i$-th feature | Captures the marginal effect of feature $x_i$ on $y$, holding all other features constant |
| $x_i$ | The $i$-th input feature | One of $n$ independent variables used to predict $y$ |
| $n$ | Total number of features | The dimensionality of the input space |
| $b$ | **Bias** (intercept) term | The predicted $y$ when all features are zero; the baseline prediction |

Or in vector notation:

$$
h_\theta(x) = \theta^T x
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $h_\theta(x)$ | **Hypothesis function** — predicted output in vector form | The linear model's prediction expressed as a compact dot product; the standard notation for multivariate regression |
| $\theta$ | Parameter vector $[\theta_0, \theta_1, \dots, \theta_n]^T$ | Contains all model parameters including bias $\theta_0$ and feature weights $\theta_1, \dots, \theta_n$ |
| $x$ | Feature vector augmented with 1 for bias | Typically $[1, x_1, x_2, \dots, x_n]^T$ so that $\theta^T x = \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n$ |
| $\theta^T x$ | **Dot product** of parameter and feature vectors | Computes the weighted sum efficiently using linear algebra |

### Variable Definitions

| Variable | Definition | Example Context |
| :--- | :--- | :--- |
| **$y$** | Predicted Target | House Price |
| **$x_1, x_2, \dots$** | Features | Area, Bedrooms, Age |
| **$m_1, m_2, \dots$** | Coefficients (Weights) | Price per sq ft, Price per bedroom |
| **$b$** | Intercept (Bias) | Base price of land |

---

## 3. Data Preprocessing: Handling Missing Values
Real-world datasets often have holes (NaN values). Before training, we must handle these using **Imputation**.

* **Strategy:** Replace missing values with the **Median** (robust to outliers) or **Mean** of the column.
* **Why?** Algorithms like Linear Regression cannot handle mathematical operations on `null` values.

---

## 4. Mathematical Deep Dive
To understand the high-dimensional matrix algebra and the pseudo-inverse engine:
* [Lecture 21: Least Squares & Over-determined Systems](../../00-Mathematics-Foundation/Lecture%2021%20Least%20Square%20Approximation%20and%20Minimum%20Norm%20Solution.md)
* [Lecture 22: Multiple Regression Matrix Form](../../00-Mathematics-Foundation/Lecture%2022%20Linear%20and%20Multiple%20Regression.md) (The Hyperplane Math)

---
**External Exercise:** [Codebasics Multivariate Regression Lab](https://github.com/codebasics/py/blob/master/ML/2_linear_reg_multivariate/2_linear_regression_multivariate.ipynb)
