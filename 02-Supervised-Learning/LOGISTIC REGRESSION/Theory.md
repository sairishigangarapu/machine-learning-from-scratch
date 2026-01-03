# Logistic Regression & Classification Metrics 📊

## 1. Concept Overview
**Logistic Regression** is a supervised learning algorithm used for **Classification** problems (predicting categorical outcomes), not regression.

### The Core Mechanism
Unlike Linear Regression, which predicts a continuous value ($y \in \mathbb{R}$), Logistic Regression predicts the **probability** ($P \in [0, 1]$) that a given input belongs to a specific class.

---

## 2. Binary Classification (2 Classes)

Used when the target has only two possible states (e.g., Spam/Not Spam, Yes/No).

### The Sigmoid Function
To map predictions to a probability between 0 and 1, we pass the linear equation ($z = mx + b$) through the **Sigmoid Activation Function**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

* If $\sigma(z) \ge 0.5 \rightarrow$ Class 1
* If $\sigma(z) < 0.5 \rightarrow$ Class 0



---

## 3. Multiclass Classification (>2 Classes)

Used when the target has 3+ categories (e.g., Digits 0-9, Fruit Types).

### Strategy 1: One-vs-Rest (OvR)
Trains $N$ binary classifiers (e.g., "Is this an Apple?" vs. "Everything else").

### Strategy 2: Softmax (Multinomial)
Generalizes the Sigmoid function to outputs probabilities for $K$ classes that sum to 1.

$$
P(y=i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
$$

---

## 4. Evaluation: The Confusion Matrix

Accuracy alone is often misleading (especially in imbalanced datasets). The **Confusion Matrix** provides a granular breakdown of errors.

| | **Predicted: 1** | **Predicted: 0** |
| :--- | :--- | :--- |
| **Actual: 1** | True Positive (TP) | False Negative (FN) |
| **Actual: 0** | False Positive (FP) | True Negative (TN) |

### Key Metrics
* **Precision:** $\frac{TP}{TP + FP}$ (Accuracy of positive predictions)
* **Recall:** $\frac{TP}{TP + FN}$ (Coverage of actual positives)
* **F1-Score:** Harmonic mean of Precision and Recall.

---
**External Exercise:** [Codebasics Logistic Regression](https://github.com/codebasics/py/tree/master/ML/7_logistic_reg)
