## Linear and Multiple Regression

*Essential Mathematics for ML — Advanced Structured Notes*

---

## 1. What is Regression?

**Regression** is the problem of learning the relationship between:
- **Input variables** (independent variables): $X_1, X_2, \ldots, X_n$ — may be qualitative or quantitative
- **Output variable** (dependent variable): $Y$ — must be **quantitative**

Mathematically, we seek a function $f$ such that:

$$
Y = f(X_1, X_2, \ldots, X_n) + \varepsilon
$$

where $\varepsilon$ represents noise or model error.

| Relationship | Model Type |
|---|---|
| $f$ is linear in the inputs | **Linear Regression** |
| $f$ is nonlinear | **Nonlinear Regression** |

---

## 2. Simple Linear Regression (The Straight Line)

When there is a **single input variable** $X$, the model is:

$$
Y = \alpha_0 + \alpha_1 X + \varepsilon
$$

where $\alpha_1$ is the **slope** (the "weight" or "annual raise") and $\alpha_0$ is the **$Y$-intercept** (the "starting point").

### Intuition
Given $m$ data points $\{(x_i, y_i)\}_{i=1}^{m}$, we find $\alpha_0$ and $\alpha_1$ that minimise the total squared error. Once the line is fit, we can **predict** $Y$ for any new $X$ value.

**Example:** If $X = \text{age}$ and $Y = \text{salary}$, $\alpha_0$ is your salary with zero experience, and $\alpha_1$ is how much your salary grows each year.

---

## 3. Multiple Linear Regression

When there are **$k$ input variables** $X_1, X_2, \ldots, X_k$, the model generalises to:

$$
Y = \alpha_0 + \alpha_1 X_1 + \alpha_2 X_2 + \cdots + \alpha_k X_k
$$

### Matrix Formulation

For $n$ data points, each equation becomes one row:

$$
y_i = \alpha_0 + \alpha_1 x_{1i} + \alpha_2 x_{2i} + \cdots + \alpha_k x_{ki}, \quad i = 1, 2, \ldots, n
$$

Stacking all $n$ equations:

$$
A\boldsymbol{\alpha} = \mathbf{y}
$$

where:

$$
A =
\begin{bmatrix}
1 & x_{11} & x_{21} & \cdots & x_{k1} \\
1 & x_{12} & x_{22} & \cdots & x_{k2} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{1n} & x_{2n} & \cdots & x_{kn}
\end{bmatrix}, \quad
\boldsymbol{\alpha} =
\begin{bmatrix}
\alpha_0 \\
\alpha_1 \\
\vdots \\
\alpha_k
\end{bmatrix}, \quad
\mathbf{y} =
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n
\end{bmatrix}
$$

Since $n \gg k+1$ in practice (many observations, few parameters), this is an **over-determined system**. The least square solution from Lecture 21 applies directly:

$$
\boxed{\boldsymbol{\alpha} = (A^T A)^{-1} A^T \mathbf{y} = A^{+} \mathbf{y}}
$$

---

## 4. Calculus Approach

Equivalently, define the error (sum of squared residuals):

$$
E = \sum_{i=1}^{n} \left(y_i - \alpha_0 - \alpha_1 x_{1i} - \cdots - \alpha_k x_{ki}\right)^2
$$

Apply the first-order conditions:

$$
\frac{\partial E}{\partial \alpha_0} = 0, \quad \frac{\partial E}{\partial \alpha_1} = 0, \quad \ldots, \quad \frac{\partial E}{\partial \alpha_k} = 0
$$

This yields $k+1$ linear equations in $k+1$ unknowns — exactly the **Normal Equations** $A^T A \boldsymbol{\alpha} = A^T \mathbf{y}$.

---

## 5. Evaluation Metrics

Given the fitted model, compute predicted values $\hat{y}_i$ for each training point $x_i$.

### Mean Squared Error (MSE)

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

### Root Mean Squared Error (RMSE)

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

### $R^2$ Score (Coefficient of Determination)

$$
R^2 = 1 - \frac{\displaystyle\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\displaystyle\sum_{i=1}^{n}(y_i - \bar{y})^2} = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}
$$

where $\bar{y} = \frac{1}{n}\sum y_i$ is the mean of the observed values.

| $R^2$ value | Interpretation |
|---|---|
| $R^2 = 1$ | Perfect fit — line passes through all points |
| $R^2 = 0$ | Model is no better than predicting the mean $\bar{y}$ |
| $0 < R^2 < 1$ | Partial fit; context-dependent threshold for "good" |

> **The Wizardry Check:** $R^2$ measures what percentage of the data's "movement" your model actually understands. $R^2 = 1$ means you've captured everything; $R^2 = 0$ means you're just guessing the average.

---

## 6. Polynomial Regression

When data does not follow a linear trend, we fit a **polynomial**:

$$
Y = \alpha_0 + \alpha_1 X + \alpha_2 X^2 + \cdots + \alpha_d X^d
$$

The mathematical process is identical to multiple regression — replace the feature columns $X_j$ with powers $X^j$. The design matrix becomes:

$$
A =
\begin{bmatrix}
1 & x_1 & x_1^2 & \cdots & x_1^d \\
1 & x_2 & x_2^2 & \cdots & x_2^d \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_n & x_n^2 & \cdots & x_n^d
\end{bmatrix}
$$

### Effect of Degree $d$

| Degree | Behaviour |
|---|---|
| $d = 1$ | Linear — may underfit curved data |
| $d = 5$ (example) | Captures the trend; small residual, generalises well |
| $d = n - 1$ | **Overfitting** — passes through every training point, $R^2 = 1$, but fails on new data |

### The Complexity Trap (Bias-Variance Tradeoff)
*   **Low Degree ($d=1$):** Too simple. It misses the curve entirely. This is **Underfitting** (High Bias).
*   **High Degree ($d=20$):** Too complex. It fits every single noise point in the training data, but it will fail miserably on any new data. This is **Overfitting** (High Variance).

---

## 7. Overfitting

**Overfitting** occurs when the model is too complex: it memorises the training data (training error $\to 0$) but generalises poorly to unseen data.

- **Symptom:** Very high $R^2$ on training data, poor predictions on test data.
- **Cause:** Model degree (or complexity) is too high relative to the amount of training data.
- **Remedy:** Regularization (see Section 8).

---

## 8. Regularization

Regularization adds a **penalty term** to the loss function to prevent excessively large coefficients, thereby controlling model complexity.

---

### 8.1 Ridge Regression (L2 Regularization)

Minimise the **penalised** residual sum of squares:

$$
\min_{\boldsymbol{\beta}} \left[ \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2 \right]
$$

The extra term $\lambda \|\boldsymbol{\beta}\|_2^2$ **shrinks** all $\beta_j$ toward zero but never forces any of them to be exactly zero.

| Parameter | Effect |
|---|---|
| $\lambda = 0$ | Ordinary least squares (no regularization) |
| $\lambda \to \infty$ | All $\beta_j \to 0$ (extreme shrinkage) |

**Key property:** Ridge produces a **dense** (non-sparse) solution — all $p$ predictors remain in the model.

---

### 8.2 Lasso Regression (L1 Regularization)

Minimise the **L1-penalised** loss:

$$
\min_{\boldsymbol{\beta}} \left[ \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j| \right]
$$

The L1 penalty can force some $\beta_j$ to be **exactly zero** when $\lambda$ is sufficiently large.

| Ridge ($\ell_2$) | Lasso ($\ell_1$) |
|---|---|
| $\beta_j$ shrink toward zero | Some $\beta_j$ become exactly zero |
| All predictors kept | Automatic **feature selection** |
| Non-sparse solution | **Sparse** solution |

**Geometric intuition:**
- Ridge constraint region: sphere (smooth, optimal solution is on the interior → all $\beta_j \neq 0$)
- Lasso constraint region: diamond with corners on axes (optimal solution often hits a corner → $\beta_j = 0$ for some $j$)

---

## 9. Summary

| Method | Model | Key Property |
|---|---|---|
| Simple Linear Regression | $Y = \alpha_0 + \alpha_1 X$ | One predictor |
| Multiple Linear Regression | $Y = \alpha_0 + \sum \alpha_j X_j$ | Multiple predictors |
| Polynomial Regression | $Y = \sum \alpha_j X^j$ | Non-linear fit |
| Ridge Regression | LS + $\lambda\|\boldsymbol{\beta}\|_2^2$ | Shrinks all $\beta$; no sparsity |
| Lasso Regression | LS + $\lambda\|\boldsymbol{\beta}\|_1$ | Sparsity; feature selection |

All methods reduce to solving an over-determined linear system via the **least square pseudo-inverse** $\boldsymbol{\alpha} = (A^T A)^{-1} A^T \mathbf{y}$ covered in Lecture 21.

**Conclusion:** You have now completed the foundation of Linear Models. You understand how to find the "best fit" for any dataset using the power of the Pseudo-Inverse. Next, we step into the world of **Recursive and Iterative optimization**!

---

### Practical Application
- **Supervised Learning Lab (Simple):** [linear_regression_lab.py](file:///home/sairishi/Sai_Rishi/GitClonedRepos/machine-learning-from-scratch/02-Supervised-Learning/LINEAR%20REGRESSION/linear_regression_lab.py)
- **Supervised Learning Lab (Multiple Variables):** [multivariate_regression_lab.py](file:///home/sairishi/Sai_Rishi/GitClonedRepos/machine-learning-from-scratch/02-Supervised-Learning/LINEAR%20REGRESSION%20WITH%20MULTIPLE%20VARIABLES/multivariate_regression_lab.py)
- **Core Concept Link:** Deep dive into the [Bias-Variance Tradeoff](file:///home/sairishi/Sai_Rishi/GitClonedRepos/machine-learning-from-scratch/01-Core-Concepts/Bias-Variance-Tradeoff.md) to understand why $L_1/L_2$ regularization is needed.
