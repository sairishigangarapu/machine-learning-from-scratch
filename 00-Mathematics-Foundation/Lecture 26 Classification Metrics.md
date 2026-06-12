## Classification Metrics

*Evaluating the Performance of Classifiers — Structured Notes*

---

## 0. Key Terminology — What All These Words Mean

### Classification

A branch of supervised learning where the goal is to assign a **class label** to an input. If there are two classes, it's **binary classification** (spam / not spam, positive / negative). If there are more than two, it's **multiclass classification** (cat / dog / bird).

### Classifier

Any algorithm that takes a data point and predicts which class it belongs to. Examples: Logistic Regression, SVM, Random Forest, Neural Networks.

### Ground Truth (Actual Label)

The true class label that we know from the data (e.g., confirmed by a reliable test like RT-PCR for COVID). This is what we compare our predictions against.

### Prediction (Predicted Label)

The class label our model *thinks* is correct. Comparing predictions to ground truth is how we evaluate performance.

### True Positive (TP)

A sample that is actually positive AND is predicted as positive. A correct hit. Example: A COVID-positive patient correctly flagged by the model.

### True Negative (TN)

A sample that is actually negative AND is predicted as negative. A correct rejection. Example: A healthy person correctly cleared by the model.

### False Positive (FP) — Type I Error

A sample that is actually negative but is predicted as positive. A false alarm. Example: A healthy person told they have COVID. Costly when it leads to unnecessary treatment or anxiety.

### False Negative (FN) — Type II Error

A sample that is actually positive but is predicted as negative. A dangerous miss. Example: A COVID-positive patient sent home to spread the virus. Usually more costly than FP in medical settings.

### Imbalanced Classes

When one class has significantly more samples than the other(s). Example: 99.9% legitimate emails vs. 0.1% spam. Accuracy becomes a misleading metric because predicting everything as "legitimate" gives 99.9% accuracy but catches zero spam.

### Precision

Out of all samples predicted as positive, how many were actually positive? Focuses on **trusting your positive predictions**.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

### Recall (Sensitivity / True Positive Rate)

Out of all actually positive samples, how many did we catch? Focuses on **not missing positives**.

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

### F1 Score

The **harmonic mean** of precision and recall. A single number that balances both. Unlike the arithmetic mean, it punishes extreme imbalance — if either precision or recall is 0, F1 is 0.

$$
F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

### Harmonic Mean

A type of average that gives more weight to small values. Used in F1 because it harshly penalizes situations where one metric is high but the other is very low.

### Support

The number of actual samples in each class. Used to compute weighted averages — classes with more samples get more influence on the overall score.

### Macro Average

Average the metric across all classes, **treating every class equally** regardless of how many samples it has. Useful when all classes are equally important.

### Weighted Average

Average the metric across all classes, **weighting by the number of samples** in each class (the support). More honest when classes are imbalanced.

---

## 1. Why Do We Need Classification Metrics?

### Motivation and Intuition

Suppose you have a classification problem — say, detecting whether a chest X-ray shows COVID-19. You try three different algorithms: Logistic Regression, Support Vector Machine, and Random Forest. All three spit out predictions, but they give different results. How do you decide which one is actually *good*?

You need a systematic way to measure performance — that's exactly what **classification metrics** are for.

### The Big Picture

Classification is a branch of supervised learning. You have labeled data (each sample has a known class), and you train a model to predict those labels. There are two flavors:

- **Binary Classification:** Exactly two classes. Positive vs. Negative, 0 vs. 1, Spam vs. Not Spam, COVID-positive vs. COVID-negative.
- **Multiclass Classification:** More than two classes. Good / Average / Bad, or classifying a player's sport: Cricket / Hockey / Football.

Some real-world classification applications: face recognition, YouTube video categorization, content moderation, medical diagnosis, text classification, sentiment analysis on Twitter, and social media analysis.

---

## 2. Confusion Matrix — The Foundation

### Motivation and Intuition

Before we can talk about accuracy, precision, or recall, we need a single table that lays out *exactly* how our model is messing up. That's the confusion matrix. It's a tabular visualization of the model's predictions versus the actual ground-truth labels.

Each **row** represents the instances in an *actual* class. Each **column** represents the instances in a *predicted* class.

### Formal Definition: Binary Case

Consider a COVID-19 test scenario. We have 1,100 patients — 100 are actually positive, 1,000 are actually negative (confirmed by RT-PCR). Now we build a classifier that predicts COVID status from chest X-ray images.

Suppose the model gives these results:
- Out of the 100 positive patients, **90** are correctly predicted as positive.
- Out of the 1,000 negative patients, **940** are correctly predicted as negative.

That leaves us with four critical numbers:

|                      | Predicted Positive | Predicted Negative |
|----------------------|--------------------|--------------------|
| **Actual Positive**  | True Positive (TP) = 90 | False Negative (FN) = 10 |
| **Actual Negative**  | False Positive (FP) = 60 | True Negative (TN) = 940 |

- **True Positive (TP):** Actual positive, predicted positive. (90)
- **True Negative (TN):** Actual negative, predicted negative. (940)
- **False Positive (FP):** Actual negative, predicted positive. (60 — these are the "false alarms")
- **False Negative (FN):** Actual positive, predicted negative. (10 — these are the dangerous misses)

```python
import numpy as np
from sklearn.metrics import confusion_matrix

y_actual = np.array(['C', 'H', 'F', 'H', 'H', 'C', 'C', 'F', 'H'])
y_pred   = np.array(['C', 'C', 'F', 'H', 'H', 'C', 'F', 'F', 'F'])

cm = confusion_matrix(y_actual, y_pred, labels=['C', 'H', 'F'])
print(cm)
```

For a multiclass problem with $n$ classes, the confusion matrix is $n \times n$.

---

## 3. Classification Accuracy

### Definition

The simplest metric: **what fraction of predictions did we get right?**

$$
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}} \times 100
$$

For our COVID example:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{90 + 940}{90 + 940 + 60 + 10} = \frac{1030}{1100} \approx 93.6\%
$$

```python
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_actual, y_pred)
print(f"Accuracy: {acc:.2%}")
```

### Failure Mode: Imbalanced Classes

Here's the trap. Suppose your dataset has 1,000 negative samples and only 100 positive samples. A brain-dead classifier that predicts **everything as negative** will still score:

$$
\text{Accuracy} = \frac{1000}{1100} \approx 90.9\%
$$

That's over 90% accuracy for a model that learned *literally nothing*. It never identifies a single positive case. This is why accuracy alone is dangerously misleading when your classes are imbalanced — a common scenario in fraud detection, rare disease diagnosis, and spam filtering.

> **Check your intuition:** If 99% of your emails are legitimate and your spam filter classifies everything as "Not Spam", what's its accuracy? *(Answer: 99%. But it's completely useless — it catches zero spam.)*

---

## 4. Precision

### Definition

Precision answers the question: **Of all the samples we *predicted* as positive, how many were actually positive?**

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

In our COVID example:

$$
\text{Precision} = \frac{90}{90 + 60} = \frac{90}{150} = 0.6
$$

Precision is your friend when **false positives are expensive**. If you're classifying emails as "Important", a false positive (flagging a casual email as urgent) is annoying but tolerable. But if you're classifying tumors as "Malignant", a false positive causes unnecessary patient trauma and invasive biopsies — you want high precision.

---

## 5. Recall (Sensitivity / True Positive Rate)

### Definition

Recall answers the question: **Of all the actually positive samples, how many did we correctly catch?**

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

In our COVID example:

$$
\text{Recall} = \frac{90}{90 + 10} = \frac{90}{100} = 0.9
$$

Recall is your friend when **false negatives are expensive**. In COVID-19 detection, a false negative means sending an infected person home to spread the virus. You want recall as close to 1.0 as possible, even if it means flagging some healthy people for follow-up testing.

### The Precision-Recall Trade-off

There's a fundamental tension here. Increasing precision typically lowers recall, and vice versa.

- If you set a very high threshold for "positive", you'll only predict positive when you're extremely sure. High precision, but you'll miss many real positives (low recall).
- If you set a low threshold, you'll catch almost all positives (high recall), but you'll also get many false positives (low precision).

You can't maximize both simultaneously. The right balance depends entirely on your problem.

---

## 6. F1 Score

### Definition

The F1 score is the **harmonic mean** of precision and recall. It gives you a single number that balances both.

$$
F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

For our COVID example:

$$
F_1 = 2 \times \frac{0.6 \times 0.9}{0.6 + 0.9} = 2 \times \frac{0.54}{1.5} = 0.72
$$

### Why Harmonic Mean?

The harmonic mean is harsh on extreme values. Consider a model with Precision = 1.0 and Recall = 0.0:

- **Arithmetic mean:** $\frac{1.0 + 0.0}{2} = 0.5$
- **F1 (harmonic mean):** $2 \times \frac{1.0 \times 0.0}{1.0 + 0.0} = 0.0$

The harmonic mean correctly punishes the model — if either precision or recall is zero, the F1 score is zero. This forces you to care about both.

F1 score favors classifiers that have **similar** precision and recall, naturally encouraging a balanced trade-off.

---

## 7. Python Implementation

Let's put it all together with a concrete example using `sklearn`.

```python
import numpy as np
from sklearn.metrics import (confusion_matrix, 
                             classification_report, 
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)

# Actual classes: 9 players and their sports
y_actual = np.array(['C', 'H', 'F', 'H', 'H', 'C', 'C', 'F', 'H'])
# C = Cricket, H = Hockey, F = Football

# Predicted classes from our (hypothetical) classifier
y_pred = np.array(['C', 'C', 'F', 'H', 'H', 'C', 'F', 'F', 'F'])

# --- Confusion Matrix ---
cm = confusion_matrix(y_actual, y_pred, labels=['C', 'H', 'F'])
print("Confusion Matrix:")
print(cm)
# Rows: actual, Columns: predicted
# C: 2 correct, 1 misclassified as F
# H: 1 correct, 1 misclassified as C, 2 misclassified as F
# F: 1 correct, 1 misclassified

# --- Classification Report (Precision, Recall, F1 per class) ---
print("\nClassification Report:")
print(classification_report(y_actual, y_pred, labels=['C', 'H', 'F']))
```

### Interpreting the Output

For the **Cricket** class:
- **Precision:** How many of the players we *predicted* as Cricket were actually Cricket?
- **Recall:** How many of the actual Cricket players did we correctly catch?
- **F1:** Harmonic mean of both.

The classification report also gives us:
- **Macro Average:** Average metric across all classes (treats all classes equally).
- **Weighted Average:** Average weighted by the number of samples in each class (more honest when classes are imbalanced).

```
              precision    recall  f1-score   support

           C       0.50      0.67      0.57         3
           H       0.50      0.25      0.33         4
           F       0.33      0.50      0.40         2

    accuracy                           0.44         9
   macro avg       0.44      0.47      0.43         9
weighted avg       0.46      0.44      0.43         9
```

### Quick Summary of the Metrics

```python
# You can also pull individual metrics directly:
acc    = accuracy_score(y_actual, y_pred)
prec   = precision_score(y_actual, y_pred, average='weighted')
rec    = recall_score(y_actual, y_pred, average='weighted')
f1     = f1_score(y_actual, y_pred, average='weighted')

print(f"Accuracy:  {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall:    {rec:.2f}")
print(f"F1 Score:  {f1:.2f}")
```

> **Check your intuition:** If your model predicts *every single sample as the majority class*, what happens to precision, recall, and F1 for the minority class? *(Answer: Precision becomes undefined (0/0), recall becomes 0, and F1 becomes 0 — the classification report will show a warning because that class has zero predicted samples.)*

---

## 8. Summary — Choosing the Right Metric

| If you care about...                              | Use this metric |
|---------------------------------------------------|-----------------|
| Overall correctness (balanced classes)            | Accuracy        |
| Avoiding false alarms (FP is expensive)           | Precision       |
| Not missing positives (FN is expensive)           | Recall          |
| A balanced trade-off between precision and recall | F1 Score        |

**The golden rule:** Always look at more than one metric. Accuracy alone lies. Precision and Recall together tell the real story. F1 gives you a single number to compare models, but always understand *why* one model has a higher F1 before trusting it.

---

### Further Reading

- **[Lecture 24: Logistic Regression-I](Lecture%2024%20Logistic%20Regression-I.md)** — Theory of the Sigmoid and MLE
- **[Lecture 25: Logistic Regression-II](Lecture%2025%20Logistic%20Regression-II.md)** — Python implementation of Logistic Regression
- **[Supervised Learning Lab](../02-Supervised-Learning/LOGISTIC%20REGRESSION/Theory.md)** — Practical lab exercises
