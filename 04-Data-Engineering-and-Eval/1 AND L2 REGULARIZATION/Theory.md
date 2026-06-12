# Regularization: Lasso (L1) & Ridge (L2)

## 1. The Core Problem: Overfitting vs. Underfitting
In Machine Learning, we aim for the **Goldilocks Zone**: a model that is not too simple (Underfitting) and not too complex (Overfitting).

### Underfitting (High Bias)
* **The Lazy Student:** Barely studies. Fails practice tests *and* the real exam.
* **Technical:** The model is too simple to capture the underlying trend (e.g., fitting a straight line to curved data).
* **Symptoms:** High Training Error, High Test Error.

### Overfitting (High Variance)
* **The Memorizer:** Memorizes every typo in the textbook. Fails the exam when questions are slightly different.
* **Technical:** The model learns the **noise** rather than the signal. It fits the training data perfectly but fails to generalize.
* **Symptoms:** Low Training Error, **High** Test Error.

---

## 2. The Solution: Regularization
Regularization prevents overfitting by **penalizing complex models**. We add a "Penalty Term" to the Loss Function (MSE) that forces the model to keep its weights ($\theta$) small.

$$
\text{Total Cost} = \text{MSE} + \text{Penalty}
$$

### A. L1 Regularization (Lasso)
**"Least Absolute Shrinkage and Selection Operator"**
* **Mechanism:** Adds the **Absolute Value** of coefficients as a penalty.
* **Formula:** $\lambda \sum |\theta_i|$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\lambda$ | Regularization strength (hyperparameter) | Controls penalty size — $\lambda = 0$ means no regularization; larger $\lambda$ = stronger penalty |
| $\theta_i$ | Model weight/parameter for feature $i$ | The coefficient the model is learning for the $i$-th feature |
| $\|\theta_i\|$ | Absolute value of the weight | L1 penalty — pushes weights toward zero, some exactly to zero |
| $\sum$ | Sum over all weights | Total penalty across all features |

* **Superpower:** **Feature Selection**. Lasso can shrink coefficients completely to **zero**, effectively removing useless features.
* **Use Case:** When you have many features, and you suspect only a few are important (Sparse data).

### B. L2 Regularization (Ridge)
* **Mechanism:** Adds the **Squared Magnitude** of coefficients as a penalty.
* **Formula:** $\lambda \sum \theta_i^2$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\lambda$ | Regularization strength (hyperparameter) | Same as L1 — controls penalty size |
| $\theta_i$ | Model weight/parameter for feature $i$ | Same as L1 |
| $\theta_i^2$ | Squared weight | L2 penalty — penalizes large weights more heavily than small ones |
| $\sum$ | Sum over all weights | Total penalty across all features |

* **Superpower:** **Weight Shrinkage**. Ridge shrinks coefficients *near* zero but rarely *exactly* to zero. It distributes the error among all features.
* **Use Case:** When most features are likely relevant and you want to prevent any single feature from dominating.

---

## 3. The Bias-Variance Tradeoff

* **High Model Complexity** $\rightarrow$ Low Bias, High Variance (Overfitting).
* **Low Model Complexity** $\rightarrow$ High Bias, Low Variance (Underfitting).
* **Regularization** helps us tune this trade-off to minimize **Total Error**.
---
<img width="1241" height="493" alt="image" src="https://github.com/user-attachments/assets/79b8ade5-4979-42d7-9678-69d14fd22b14" />
---

**External Exercise:** [Codebasics L1/L2 Regularization Lab](https://github.com/codebasics/py/blob/master/ML/16_regularization/16_regularization.ipynb)
