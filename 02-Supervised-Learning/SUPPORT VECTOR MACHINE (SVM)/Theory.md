# Support Vector Machine (SVM) ⚔️

## 1. Concept Overview
**Support Vector Machine (SVM)** is a powerful supervised learning algorithm used primarily for classification. Ideally, it produces a significant accuracy with less computation power.

### The Core Goal: Maximize the Margin
SVM distinguishes itself from other classifiers by finding the **best** boundary (Hyperplane) that separates the classes.
* **Hyperplane:** The line (2D), plane (3D), or hyper-surface (nD) that splits the data.
* **Margin:** The distance between the hyperplane and the nearest data points.
* **Support Vectors:** The specific data points closest to the hyperplane. These are the "load-bearing" points; if you remove other points, the boundary doesn't change.

> **Objective:** Find the hyperplane that maximizes the margin. Wider margin = Better generalization.

---

## 2. Key Hyperparameters

### A. Regularization ($C$)
Controls the trade-off between a smooth decision boundary and classifying training points correctly.
* **High $C$:** Strict. Tries to classify every training point correctly. Risk of **Overfitting**.
* **Low $C$:** Lenient. Accepts some misclassifications to maintain a wider margin. Better **Generalization**.

### B. Gamma ($\gamma$)
Defines how far the influence of a single training example reaches.
* **High $\gamma$:** Only close points influence the boundary. Result: Wobbly, complex curves (Overfitting).
* **Low $\gamma$:** Far points influence the boundary. Result: Straighter, smoother lines.

---

## 3. The Kernel Trick 🥜
What if data isn't linearly separable (e.g., a circle inside another circle)?
SVM uses **Kernels** to project data into a higher-dimensional space where it *becomes* linearly separable.

| Kernel | Use Case |
| :--- | :--- |
| **Linear** | Simple text classification, linearly separable data. |
| **RBF (Radial Basis Function)** | The default. Good for non-linear, complex boundaries. |
| **Polynomial** | Image processing, curved boundaries. |

---

## 4. Hard Margin vs. Soft Margin

### Hard Margin SVM
Requires data to be **perfectly linearly separable**. Maximizes the margin with zero misclassifications. Sensitive to outliers.

### Soft Margin SVM
Allows some misclassifications by introducing **slack variables** $\xi_i$:

$$
\min \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i
$$

The parameter $C$ controls the penalty for misclassification:
* $C = \infty$ → Hard Margin (no errors allowed)
* $C \to 0$ → Wide margin, more errors tolerated

> **In practice, always use soft margin** — real-world data is rarely perfectly separable.

---

## 5. Code Example

```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Compare kernels
for kernel in ['linear', 'rbf', 'poly']:
    svm = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    acc = svm.score(X_test, y_test)
    n_sv = svm.n_support_.sum()
    print(f"  {kernel:8s} — Accuracy: {acc:.3f}, Support Vectors: {n_sv}")
```

---

## 6. Support Vectors: Why They Matter

Only the support vectors (points closest to the decision boundary) determine the hyperplane. All other points can be removed without changing the boundary. This makes SVM:
* **Memory efficient** — only stores support vectors.
* **Robust to outliers** — far-away points don't influence the boundary (if $C$ is reasonable).

---

**External Exercise:** [Codebasics SVM Lab](https://github.com/codebasics/py/blob/master/ML/10_svm/Exercise/10_svm_exercise_digits.ipynb)
