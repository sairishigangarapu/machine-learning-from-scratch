## Principal Component Analysis (PCA)

*Essential Mathematics for ML — Advanced Structured Notes*

---

## 1. Introduction to PCA

### Motivation and Intuition
Imagine a dataset of 3D points floating in a room, but they happen to be structured exactly like a flat pancake. Even though the data uses 3 numbers $(x, y, z)$ to describe each point, practically all the variance (the "signal") lies along the two flat dimensions of the pancake. The thickness of the pancake (the third dimension) represents pure experimental noise.

Principal Component Analysis (PCA) gives us a way to computationally construct a new coordinate system aligned perfectly with the flat face of the pancake. We then discard the dimension pointing into the thickness. We have just reduced our data from 3D to 2D without losing any meaningful signal.

In Machine Learning, real-world datasets often have thousands of dimensions (e.g., pixel intensities of an image). Feeding all this raw data into models causes the **"Curse of Dimensionality"**, leading to massive overfitting. PCA distills high-dimensional chaos into its most information-dense underlying factors.

### Formal Definition
PCA is an orthogonal linear transformation that projects the data into a new coordinate system such that the greatest variance by some scalar projection of the data comes to lie on the first coordinate (the first principal component), the second greatest on the second coordinate, and so on.

---

## 2. Statistical Preliminaries

To understand how PCA identifies these "best directions", we define statistical measures:

* **Mean ($\mu$):** The central tendency of a feature.

$$
\mu = \frac{1}{n} \sum_{i=1}^{n} \mathbf{x}_i
$$

* **Variance/Standard Deviation:** Measures the spread of a single feature.
* **Covariance ($\text{cov}(X, Y)$):** Measures how two variables change dynamically together.

$$
\sigma_{XY} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu_X)(y_i - \mu_Y)
$$

---

## 3. The Covariance Matrix ($\Sigma$)

For a dataset with $d$ features, the covariance matrix $\Sigma$ is a $d \times d$ symmetric matrix. It is the mathematical heart of PCA because its structure captures the full 3D shape and orientation of our metaphorical pancake.

### Matrix Structure
Diagonal elements are the **variances**, off-diagonals are **covariances**:

$$
\Sigma = \begin{bmatrix} 
\text{var}(X_1) & \text{cov}(X_1, X_2) & \dots & \text{cov}(X_1, X_d) \\
\text{cov}(X_2, X_1) & \text{var}(X_2) & \dots & \text{cov}(X_2, X_d) \\
\vdots & \vdots & \ddots & \vdots \\
\text{cov}(X_d, X_1) & \text{cov}(X_d, X_2) & \dots & \text{var}(X_d) 
\end{bmatrix}
$$

### Algebraic Computation
If $C$ is the **centered data matrix** of size $n \times d$ (mean subtracted from columns):

$$
\Sigma = \frac{1}{n} C^T C
$$

```python
import numpy as np

# A dataset with 100 samples and 3 features
data = np.random.rand(100, 3)

# 1. Standardize (Center) the data
C = data - np.mean(data, axis=0)

# 2. Compute Covariance matrix
Sigma = (C.T @ C) / data.shape[0]

# NumPy has a built-in function that does exactly this:
# Sigma = np.cov(data, rowvar=False)
```

### Key Geometric Properties

1. **Symmetry:** $\Sigma = \Sigma^T$.
2. **Positive Semi-Definite:** $\mathbf{u}^T \Sigma \mathbf{u} \ge 0$, ensuring eigenvalues (variances) are never negative.
3. **Rotation and Variance:** The eigenvectors of $\Sigma$ point precisely in the directions of maximum variance in the data.

---

## 4. Principal Components

The **Principal Components** are the **eigenvectors** of the covariance matrix.

* **First Principal Component (PC1):** The eigenvector corresponding to the **largest eigenvalue**. This spans the longest axis of the data.
* **Subsequent Components:** PC2 is the eigenvector of the second-largest eigenvalue, mutually orthogonal to PC1.

```python
# 3. Eigendecomposition to find the components
eigenvalues, eigenvectors = np.linalg.eig(Sigma)

# 4. Sort in descending order to find PC1, PC2...
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
principal_components = eigenvectors[:, sorted_indices]
```

---

## 5. The PCA Projection Step

To compress an $n$-dimensional dataset to $k$ dimensions ($k < n$):

1. **Select Top $k$ Components:** Extract the first $k$ columns of the sorted eigenvector matrix. This forms our projection matrix $W$ ($n \times k$).
2. **Projection:** Project the centered data $C$ onto these eigenvectors: $Y = C W$.

```python
# Suppose we want to reduce from 3D to 2D
k = 2
W = principal_components[:, :k]  # Projection matrix

# 5. Project the data!
reduced_data = C @ W  # Now shaped (100, 2)
```

---

## 6. Significance and Failure Modes

* **Information Preservation:** The ratio `eigenvalue / sum(all eigenvalues)` tells us exactly what percentage of the variance is preserved by that component.
* **Failure Mode (Non-Linearity):** PCA strictly captures linear correlations. If your data forms a 3D spiral (a "Swiss Roll"), PCA will utterly fail to unroll it because a global linear plane cannot model curves. This is why we need Manifold Learning (like t-SNE or UMAP) or Autoencoders for highly non-linear data.
* **Failure Mode (Feature Scaling):** If feature $X_1$ is measured in millimeters, and $X_2$ in kilometers, $X_1$ will mathematically dominate the variance. PCA will blindly align PC1 across $X_1$, completely ruining the analysis. **Always scale/normalize features to unit variance before PCA.**

> **Check your intuition:** If a dataset is a completely uniform, perfectly spherical cloud of noise in 3D, what does PCA do? *(Answer: It does absolutely nothing useful. All eigenvalues of the covariance matrix will be identical, meaning there are no "principal" directions since the variance is equal in every direction.)*
