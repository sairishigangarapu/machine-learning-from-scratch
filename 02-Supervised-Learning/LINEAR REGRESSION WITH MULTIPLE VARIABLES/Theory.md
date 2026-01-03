# Multivariate Linear Regression 📉📈

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

Or in vector notation:

$$
h_\theta(x) = \theta^T x
$$

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
**External Exercise:** [Codebasics Multivariate Regression Lab](https://github.com/codebasics/py/blob/master/ML/2_linear_reg_multivariate/2_linear_regression_multivariate.ipynb)
