# Principal Component Analysis (PCA)

## 1. Concept Overview: Dimensionality Reduction
**Principal Component Analysis (PCA)** is an unsupervised learning technique used for **Dimensionality Reduction**. It transforms a large set of correlated variables into a smaller set of uncorrelated variables called **Principal Components**, while retaining as much of the original information (variance) as possible.

### The Goal
To reduce the number of features (compress data) to:
1. **Reduce Overfitting:** Fewer features = less noise (`Curse of Dimensionality`).
2. **Speed Up Training:** Fewer calculations for the model.
3. **Visualization:** Project high-dimensional data (e.g., 64 pixels) into 2D or 3D plots.

---

## 2. The Mathematics: Eigenvectors & Eigenvalues

PCA relies on Linear Algebra to find the "directions" of maximum variance.

### A. Variance & Covariance
* **Variance:** How spread out the data is. PCA assumes **High Variance = High Information**.
* **Covariance:** How two features vary together. PCA aims to remove this correlation.

### B. The Principal Components
1. **Eigenvectors (The Direction):** The new axes (lines) that point in the direction of the greatest variance in the data. They are orthogonal (perpendicular) to each other.
2. **Eigenvalues (The Magnitude):** The amount of variance captured by each Eigenvector.
 * High Eigenvalue $\rightarrow$ Important Component (Keep it).
 * Low Eigenvalue $\rightarrow$ Noise (Discard it).

---

## 3. The Algorithm Steps

1. **Standardization:** Scale the data (mean = 0, variance = 1). *Crucial, otherwise large-scale features dominate.*
2. **Covariance Matrix:** Compute the $n \times n$ matrix to understand feature correlations.
3. **Eigendecomposition:** Calculate Eigenvectors and Eigenvalues from the Covariance Matrix.
4. **Sort & Select:** Sort components by Eigenvalue (High to Low) and keep the top $k$ components that explain the desired variance (e.g., 95%).
5. **Projection:** Transform the original data onto these new axes.

---

## 4. Advantages vs. Disadvantages

| Pros (Why use it) | Cons (The Trade-off) |
| :--- | :--- |
| **Noise Reduction:** Removes irrelevant details. | **Loss of Interpretability:** New features (PC1, PC2) are abstract math mixtures, not real-world features like "Age" or "Income." |
| **Visualization:** Can visualize 100D data in 2D. | **Information Loss:** You always lose *some* data during compression. |
| **Efficiency:** Faster training for downstream models. | **Linear Assumption:** Only works well if data relationships are linear. |

---
**External Exercise:** [Codebasics PCA Lab](https://github.com/codebasics/py/blob/master/ML/18_pca/pca_digits.ipynb)

---

## 5. Mathematical Deep Dive
To deeply understand the linear algebra engine driving this algorithm (Covariance Matrices, Eigenvectors, Representation Error), please reference our **Mathematics Foundation** knowledge bank:
* [Lecture 09: Eigenvalues & Eigenvectors](../../00-Mathematics-Foundation/Lecture%2009%20Eigenvalues%20and%20Eigenvectors.md) (The Core Engines)
* [Lecture 12: Singular Value Decomposition](../../00-Mathematics-Foundation/Lecture%2012%20Singular%20Value%20Decomposition.md) (The Robust Implementation Engine)
* [Lecture 16: PCA](../../00-Mathematics-Foundation/Lecture%2016%20Principal%20Component%20Analysis.md) (Covariance Geometry & High-Level Intuition)
* [Lecture 17: PCA - Part II](../../00-Mathematics-Foundation/Lecture%2017%20PCA%20Derivation%20and%20Examples.md) (Optimization Proofs & Calculus)
* [Lecture 18: PCA - Python](../../00-Mathematics-Foundation/Lecture%2018%20Python%20implementation%20of%20PCA.md) (NumPy From-Scratch)
