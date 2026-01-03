# Linear Regression: Foundations 📉

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

<img width="840" height="467" alt="image" src="https://github.com/user-attachments/assets/afc9829a-d38a-4496-89a6-3ec3d4ec6d1e" />


The goal of training is to find the optimal values for $m$ and $c$ that minimize the **Mean Squared Error (MSE)**.

---
**External Exercise:** [Codebasics Linear Regression Lab](https://github.com/codebasics/py/tree/master/ML/1_linear_reg)
