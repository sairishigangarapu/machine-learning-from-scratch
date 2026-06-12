## SVM Implementation in Python

*Essential Mathematics for ML — Structured Notes*

---

## 1. Setup and Data Preparation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Generate datasets
# Dataset 1: Linearly separable
X_linear, y_linear = make_blobs(n_samples=200, centers=2, 
                                 cluster_std=1.0, random_state=42)
y_linear = np.where(y_linear == 0, -1, 1)

# Dataset 2: Non-linearly separable
X_moon, y_moon = make_moons(n_samples=200, noise=0.2, random_state=42)
y_moon = np.where(y_moon == 0, -1, 1)

# Train/test split
X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
    X_linear, y_linear, test_size=0.3, random_state=42)
X_train_moon, X_test_moon, y_train_moon, y_test_moon = train_test_split(
    X_moon, y_moon, test_size=0.3, random_state=42)

# Standardize features (important for SVM!)
scaler_lin = StandardScaler()
X_train_lin = scaler_lin.fit_transform(X_train_lin)
X_test_lin = scaler_lin.transform(X_test_lin)

scaler_moon = StandardScaler()
X_train_moon = scaler_moon.fit_transform(X_train_moon)
X_test_moon = scaler_moon.transform(X_test_moon)
```

---

## 2. Linear SVM

```python
# Linear SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_lin, y_train_lin)

print("Linear SVM Results:")
print(f"Training accuracy: {svm_linear.score(X_train_lin, y_train_lin):.4f}")
print(f"Test accuracy: {svm_linear.score(X_test_lin, y_test_lin):.4f}")
print(f"Number of support vectors: {svm_linear.n_support_}")
print(f"\nClassification Report:")
print(classification_report(y_test_lin, svm_linear.predict(X_test_lin)))
```

### Decision Boundary Visualization

```python
def plot_decision_boundary(model, X, y, title, ax):
    """Plot SVM decision boundary and margins."""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='k')
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=100, facecolors='none', edgecolors='yellow', linewidths=2)
    ax.set_title(title)
```

---

## 3. Kernel Comparison

```python
kernels = {
    'linear': SVC(kernel='linear', C=1.0),
    'poly': SVC(kernel='poly', degree=3, C=1.0),
    'rbf': SVC(kernel='rbf', gamma='scale', C=1.0),
    'sigmoid': SVC(kernel='sigmoid', gamma='scale', C=1.0),
}

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, (name, model) in zip(axes, kernels.items()):
    model.fit(X_train_moon, y_train_moon)
    plot_decision_boundary(model, X_test_moon, y_test_moon, 
                          f"{name} (acc={model.score(X_test_moon, y_test_moon):.3f})", ax)
plt.tight_layout()
plt.savefig('kernel_comparison.png', dpi=150)
```

---

## 4. Hyperparameter Tuning with Grid Search

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_moon, y_train_moon)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
print(f"Test accuracy: {grid_search.best_estimator_.score(X_test_moon, y_test_moon):.4f}")
```

---

## 5. Support Vector Analysis

```python
# Analyze support vectors
svm_best = SVC(kernel='rbf', C=10, gamma=1)
svm_best.fit(X_train_moon, y_train_moon)

print("Support Vector Analysis:")
print(f"Total support vectors: {svm_best.n_support_.sum()}")
print(f"SV per class: {svm_best.n_support_}")
print(f"Fraction of training data: {svm_best.n_support_.sum() / len(X_train_moon):.2%}")

# Support vector indices
sv_indices = svm_best.support_
print(f"\nSupport vector indices: {sv_indices}")

# Dual coefficients (alpha * y)
print(f"\nDual coefficients shape: {svm_best.dual_coef_.shape}")
print(f"First 5 dual coefficients: {svm_best.dual_coef_[0, :5]}")
```

---

## 6. Multi-class SVM

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Iris dataset (3 classes)
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
scaler = StandardScaler()
X_iris = scaler.fit_transform(X_iris)

X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42)

# One-vs-One (default) strategy
svm_multi = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_multi.fit(X_train, y_train)

print("Multi-class SVM (OvO):")
print(f"Training accuracy: {svm_multi.score(X_train, y_train):.4f}")
print(f"Test accuracy: {svm_multi.score(X_test, y_test):.4f}")
print(f"Number of classifiers: {len(svm_multi.classes_) * (len(svm_multi.classes_) - 1) // 2}")
```

---

## 7. SVM vs Other Classifiers

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Linear SVM': SVC(kernel='linear', C=1.0),
    'RBF SVM': SVC(kernel='rbf', C=10, gamma=1),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
}

print("Classifier Comparison (Moon Dataset):")
print(f"{'Classifier':<20} {'Train Acc':<12} {'Test Acc':<12} {'# SVs':<10}")
print("-" * 54)

for name, clf in classifiers.items():
    clf.fit(X_train_moon, y_train_moon)
    train_acc = clf.score(X_train_moon, y_train_moon)
    test_acc = clf.score(X_test_moon, y_test_moon)
    n_sv = getattr(clf, 'n_support_', None)
    n_sv_str = str(n_sv) if n_sv is not None else 'N/A'
    print(f"{name:<20} {train_acc:<12.4f} {test_acc:<12.4f} {n_sv_str:<10}")
```

---

## 8. Handling Imbalanced Classes

### The Problem
When one class has far more samples than the other, the SVM bias toward the majority class leads to poor recall on the minority class.

### The Solution: Class Weights
Set `class_weight='balanced'` to automatically weight classes inversely proportional to their frequency:

```python
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Create imbalanced data
X, y = make_blobs(n_samples=[200, 20], centers=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Without class weights
svm_unbalanced = SVC(kernel='rbf', C=1.0)
svm_unbalanced.fit(X_train, y_train)
print("Without class weights:")
print(classification_report(y_test, svm_unbalanced.predict(X_test)))

# With class weights
svm_balanced = SVC(kernel='rbf', C=1.0, class_weight='balanced')
svm_balanced.fit(X_train, y_train)
print("With class weights='balanced':")
print(classification_report(y_test, svm_balanced.predict(X_test)))
```

**Intuition:** The `balanced` mode assigns weight $w_c = n / (n_{\text{classes}} \times n_c)$ to each class $c$. If class 0 has 200 samples and class 1 has 20, class 1 gets 5x the weight — misclassifying a minority sample is penalized 5x more.

---

## 9. When to Use SVMs

| Scenario | Recommendation |
|:---|:---|
| High-dimensional (text, genomics) | SVM with linear kernel |
| Medium dataset, non-linear boundary | SVM with RBF kernel |
| Very large dataset ($n > 100K$) | SGDClassifier or deep learning |
| Need probabilities | Use `SVC(probability=True)` (calibration) |
| Interpretability needed | Use linear SVM (inspect weights) |
| Multi-class | Use one-vs-one or one-vs-rest |

> **Check your intuition:** Why should you standardize features before using SVM? *(Answer: SVM uses distances (via dot products) to define margins. If one feature has range [0, 1000] and another [0, 1], the first dominates the distance computation. Standardization ensures all features contribute equally.)*

---

## Prerequisites and Further Reading
- **Previous:** [Lecture 59: Kernels and the Kernel Trick](Lecture%2059%20Kernels%20and%20the%20Kernel%20Trick.md) — Theoretical foundation for non-linear SVMs
- **Related:** [Lecture 54: Introduction to Support Vector Machines](Lecture%2054%20Introduction%20to%20Support%20Vector%20Machines.md) — SVM concepts
- **Related:** [Lectures 55-59](Lecture%2055%20Maximum%20Margin%20Classification.md) through [Lecture 59: Kernels and the Kernel Trick](Lecture%2059%20Kernels%20and%20the%20Kernel%20Trick.md) — Complete SVM theory
- **Related:** [Lecture 25: Logistic Regression-II](Lecture%2025%20Logistic%20Regression-II.md) — Alternative classification approach
- **Related:** [Lecture 23: Linear and Multiple Regression-II](Lecture%2023%20Linear%20and%20Multiple%20Regression-II.md) — Regression counterpart
