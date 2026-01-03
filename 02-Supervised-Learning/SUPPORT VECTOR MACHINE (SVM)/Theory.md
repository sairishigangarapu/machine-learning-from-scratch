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
**External Exercise:** [Codebasics SVM Lab](https://github.com/codebasics/py/blob/master/ML/10_svm/Exercise/10_svm_exercise_digits.ipynb)
