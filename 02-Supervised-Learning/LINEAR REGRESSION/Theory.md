# Linear Regression: Foundations

## 1. Concept Overview
**Linear Regression** is a supervised learning algorithm used for predicting a continuous dependent variable ($y$) based on one or more independent variables ($x$).

### The Core Logic
The algorithm attempts to model the relationship between variables by fitting a linear equation to observed data. It calculates the **"Line of Best Fit"** by minimizing the offset (residuals) between the actual data points and the predicted line.

---

## 2. Mathematical Representation

The relationship is expressed as the equation of a straight line:

$$
y = mx + c
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $y$ | Dependent variable (target) | What we want to predict (e.g., house price) |
| $x$ | Independent variable (feature) | The input we use to predict (e.g., area) |
| $m$ | Slope (weight) | How much $y$ changes for each unit increase in $x$ — the "learned" parameter |
| $c$ | Intercept (bias) | The value of $y$ when $x = 0$ — the baseline prediction |

In Machine Learning notation, this is often written as:

$$
h_\theta(x) = \theta_1 x + \theta_0
$$

### Variable Definitions

| Variable | ML Term | Definition |
| :--- | :--- | :--- |
| **$y$** | Target | The dependent variable we want to predict (e.g., Price). |
| **$x$** | Feature | The independent input variable (e.g., Area). |
| **$m$ ($\theta_1$)** | Weights | The slope or gradient. It determines the impact of $x$ on $y$. |
| **$c$ ($\theta_0$)** | Bias | The $y$-intercept. The baseline value when $x = 0$. |

---

## 3. Visualization

![Linear Regression](https://github.com/user-attachments/assets/afc9829a-d38a-4496-89a6-3ec3d4ec6d1e)


The goal of training is to find the optimal values for $m$ and $c$ that minimize the **Mean Squared Error (MSE)**.

---

## 4. Mathematical Deep Dive
To understand the matrix calculus and pseudo-inverse math behind the line of best fit:
* [Lecture 21: Least Squares & Pseudo-Inverse](../../00-Mathematics-Foundation/Lecture%2021%20Least%20Square%20Approximation%20and%20Minimum%20Norm%20Solution.md) (The Algebraic Engine)
* [Lecture 22: Linear and Multiple Regression](../../00-Mathematics-Foundation/Lecture%2022%20Linear%20and%20Multiple%20Regression.md) (The Core Foundation)
* [Bias-Variance Tradeoff](../../01-Core-Concepts/Bias-Variance-Tradeoff.md) (Understanding Regularization: Ridge & Lasso)

---
**External Exercise:** [Codebasics Linear Regression Lab](https://github.com/codebasics/py/tree/master/ML/1_linear_reg)
