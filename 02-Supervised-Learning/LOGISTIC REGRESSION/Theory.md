# Logistic Regression & Classification Metrics

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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $z$ | Linear combination $\mathbf{w}^T\mathbf{x} + b$ | The raw output of the model before activation — can be any real number |
| $\sigma(z)$ | Sigmoid of $z$ | Squashes $z$ into the range $(0, 1)$ — interpretable as a probability |
| $e^{-z}$ | Exponential of negative $z$ | When $z$ is large positive, $e^{-z} \approx 0$, so $\sigma \approx 1$; when $z$ is large negative, $e^{-z} \approx \infty$, so $\sigma \approx 0$ |

* If $\sigma(z) \ge 0.5 \rightarrow$ Class 1
* If $\sigma(z) < 0.5 \rightarrow$ Class 0

> **Key Insight:** The sigmoid function squashes any real number into the range (0, 1), making it interpretable as a probability. The decision boundary is at $z = 0$ (i.e., $\mathbf{w}^T\mathbf{x} + b = 0$).

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

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $P(y=i)$ | Probability that the input belongs to class $i$ | The output for class $i$ — all class probabilities sum to 1 |
| $z_i$ | Logit (raw score) for class $i$ | The linear output for class $i$ before softmax — larger $z_i$ = higher probability |
| $e^{z_i}$ | Exponential of the logit for class $i$ | Converts any real number to a positive value — preserves ordering |
| $\sum_{j=1}^K e^{z_j}$ | Sum of exponentials across all $K$ classes | The normalization constant — ensures probabilities sum to 1 |
| $K$ | Total number of classes | The number of possible categories (e.g., 10 for digit classification) |
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

 ## 5. Mathematical Deep Dive
 Understanding the leap from regression to classification:
 * [Lecture 19 & 20: LDA](../../00-Mathematics-Foundation/Lecture%2019%20Linear%20Discriminant%20Analysis.md) (How classification works using linear separation)
 * [Lecture 22: Regression Foundations](../../00-Mathematics-Foundation/Lecture%2022%20Linear%20and%20Multiple%20Regression.md) (The linear engine $z = mx+b$)

 ---
**External Exercise:** [Codebasics Logistic Regression](https://github.com/codebasics/py/tree/master/ML/7_logistic_reg)
