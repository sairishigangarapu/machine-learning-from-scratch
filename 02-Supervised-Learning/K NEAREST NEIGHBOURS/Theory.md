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
**External Exercise:** [Codebasics KNN Lab](https://github.com/codebasics/py/blob/master/ML/17_knn_classification/17_knn_classification.ipynb)
