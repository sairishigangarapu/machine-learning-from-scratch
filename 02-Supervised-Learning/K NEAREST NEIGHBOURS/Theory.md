# K-Nearest Neighbors (KNN) 🏘️

## 1. Concept Overview
**K-Nearest Neighbors (KNN)** is a supervised learning algorithm used for both Classification and Regression. It is an **Instance-Based** or **Lazy Learning** algorithm, meaning it does not construct an internal model during training. Instead, it stores the entire training dataset and performs computation only during inference.

### The Core Logic
**"Tell me who your friends are, and I will tell you who you are."**
To classify a new data point, the algorithm identifies the $K$ closest data points (neighbors) in the feature space and assigns the class based on a majority vote.

---

## 2. How It Works

1. **Calculate Distance:** Compute the distance between the query point and every point in the dataset.
2. **Find Neighbors:** Sort the distances and select the top $K$ nearest points.
3. **Vote (Classification):** Assign the class that appears most frequently among the neighbors.
4. **Average (Regression):** Assign the average value of the neighbors' targets.

---

## 3. Key Components

### A. The Hyperparameter $K$
* **Small $K$ (e.g., 1 or 3):** Sensitive to noise/outliers. Can lead to **Overfitting** (High Variance).
* **Large $K$:** Smoothes out the decision boundary. Can lead to **Underfitting** (High Bias).
* **Tip:** Choose an odd number for $K$ to avoid tie votes in binary classification.

### B. Distance Metrics
How do we define "Close"?
* **Euclidean Distance:** (Standard) Straight-line distance.
  $$d(p, q) = \sqrt{\sum (p_i - q_i)^2}$$
* **Manhattan Distance:** Grid-like path (sum of absolute differences).
* **Minkowski Distance:** A generalization of both.

### C. Feature Scaling (Crucial!)
Since KNN relies on distance, features with larger magnitudes (e.g., Salary: 100,000) will dominate features with smaller magnitudes (e.g., Age: 25). **Normalization (MinMax) or Standardization (StandardScaler) is mandatory.**

---

## 4. Advantages vs. Disadvantages

### ✅ Pros
* **Simple:** Easy to understand and implement.
* **Non-Parametric:** Makes no assumptions about the underlying data distribution.
* **Versatile:** Works for multi-class problems naturally.

### ❌ Cons
* **Computationally Expensive:** Must calculate distance to *every* training point for *every* prediction ($O(N)$ inference).
* **Memory Intensive:** Must keep the entire dataset in RAM.
* **Curse of Dimensionality:** Performance degrades rapidly as the number of features increases (distance becomes meaningless in high dimensions).

---

## 5. Code Example

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Scaling is mandatory for KNN
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Compare different K values
for k in [1, 3, 5, 10, 20]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_s, y_train)
    acc = knn.score(X_test_s, y_test)
    print(f"  K={k:2d} — Accuracy: {acc:.3f}")
```

---

**External Exercise:** [Codebasics KNN Lab](https://github.com/codebasics/py/blob/master/ML/17_knn_classification/17_knn_classification.ipynb)
