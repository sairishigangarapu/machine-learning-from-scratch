## Lecture 25: Python Implementation of Logistic Regression

### 1. Introduction to Implementation
In this lecture, we transition from the theory of Maximum Likelihood Estimation (MLE) and Sigmoid functions to practical implementation in Python. We will explore two distinct datasets:
1. **Binary Classification:** Predicting whether a user will purchase a product based on Age and Salary.
2. **Multiclass Classification (Advanced):** Recognizing face images of famous individuals.

---

### 2. Case Study 1: User Procurement (Binary)
We use a company dataset containing info on 400 users.

#### Data Selection
Not all columns in a dataset are useful features. For this problem:
- **User ID:** Irrelevant (unique identifier).
- **Gender:** Secondary importance.
- **Age & Estimated Salary:** Primary independent variables.
- **Purchased (0 or 1):** The target variable (dependent variable).

#### Workflow
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# 1. Prepare Data
X = dataset.iloc[:, [2, 3]].values # Age and Salary
y = dataset.iloc[:, 4].values      # Purchased (0/1)

# 2. Train/Test Split (75/25)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```

---

### 3. The Necessity of Feature Scaling (Standardization)
In our data, Age $(\approx 20-50)$ and Salary $(\approx 20,000-100,000)$ have vastly different scales. In algorithms like Logistic Regression that rely on Gradient Descent, features with larger magnitudes will dominate the gradient, leading to poor convergence.

**The Solution: Z-Score Normalization**
We shift the data to have a mean ($\mu$) of 0 and a standard deviation ($\sigma$) of 1:

$$ z = \frac{x - \mu}{\sigma} $$

```python
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

---

### 4. Training and Evaluation
```python
# 1. Fit the Model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# 2. Predict on Test Data
y_pred = classifier.predict(X_test)

# 3. Evaluate via Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
# Example result: 89% accuracy
# 65 correct 0s, 24 correct 1s, 11 misclassified
```

---

### 5. Case Study 2: Face Recognition (Multiclass)
This advanced example uses the **LFW (Labeled Faces in the Wild)** dataset.
- **Features:** ~3,000 pixels per image (62x47 dimension).
- **Classes:** 8 people (e.g., Colin Powell, George W. Bush).
- **Complexity:** Higher-dimensional input requiring robust optimization.

#### Implementation Logic
```python
from sklearn.datasets import fetch_lfw_people

# fetch images with at least 60 samples per person
faces = fetch_lfw_people(min_faces_per_person=60)
X = faces.data
y = faces.target

# The same LogisticRegression object handles multiclass via
# 'one-vs-rest' (OvR) or 'multinomial' strategies internally.
model = LogisticRegression()
model.fit(X_train, y_train)
```

#### Outcome
Even in high dimensions, Logistic Regression provides a solid baseline. In this exercise, the model achieved **81% accuracy** in identifying faces across 8 classes.

---

### 6. Summary of the Week
We have now mastered the two pillars of foundational supervised learning:
1. **Linear Regression:** Modeling continuous outcomes using the Normal Equation or Least Squares.
2. **Logistic Regression:** Modeling discrete class probabilities using the Sigmoid function and Maximum Likelihood Estimation (MLE).

---

### Practical Application
- **Supervised Learning Lab:** [logistic_regression_lab.py](../02-Supervised-Learning/LOGISTIC%20REGRESSION/Theory.md)
- **Math Deep-Dive:** Review [Lecture 24 (Theory)](../../00-Mathematics-Foundation/Lecture%2024%20Logistic%20Regression-I.md) to understand the MLE logic behind `model.fit()`.
