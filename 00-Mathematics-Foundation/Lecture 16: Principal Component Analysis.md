## Principal Component Analysis (PCA)

*Essential Mathematics for ML — Advanced Structured Notes*

---

## 1. Introduction to PCA

Principal Component Analysis (PCA) is a powerful **dimensionality reduction** and manifold learning technique. In Machine Learning, we often encounter high-dimensional data (many features) where some features are redundant or noise. PCA allows us to project this data into a lower-dimensional space while preserving as much information (variance) as possible.

### Motivation Example

Imagine a dataset of cities graded on four parameters: Education ($X_1$), Transport ($X_2$), Entertainment ($X_3$), and Safety ($X_4$). Each city is a vector in $\mathbb{R}^4$.
To visualize this or simplify models, we might want to reduce this to $\mathbb{R}^3$ or $\mathbb{R}^2$. This requires a transformation matrix $W$:


$$\mathbf{y} = W\mathbf{x}$$


If we go from 4D to 3D, $W$ is a $3 \times 4$ matrix. PCA provides the optimal $W$ such that the "spread" or variation of the data is maximized in the new space.

---

## 2. Statistical Preliminaries

To understand how PCA identifies the best directions, we must define key statistical measures:

* **Mean ($\mu$):** The central tendency of the data.

$$\mu = \frac{1}{n} \sum_{i=1}^{n} \mathbf{x}_i$$


* **Standard Deviation ($\sigma$):** Measures the variability or spread of a single variable around the mean.
* **Covariance ($\text{cov}(X, Y)$):** Measures how two variables change together.

$$\sigma_{XY} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu_X)(y_i - \mu_Y)$$


* **Positive:** Variables increase/decrease together.
* **Negative:** One increases while the other decreases.
* **Zero:** Variables are independent.



---

## 3. The Covariance Matrix ($\Sigma$)

For a dataset with $k$ attributes, the covariance matrix is a $k \times k$ symmetric matrix that captures the relationships between all pairs of features:
$$\Sigma = \begin{bmatrix}
\text{var}(X_1) & \text{cov}(X_1, X_2) & \dots \
\text{cov}(X_2, X_1) & \text{var}(X_2) & \dots \
\vdots & \vdots & \ddots
\end{bmatrix}$$
If $C$ is the centered data matrix (where the mean is subtracted from each row), then:


$$\Sigma = \frac{1}{n} C^T C$$


Because $\Sigma$ is **symmetric**, it is guaranteed to have real eigenvalues and orthogonal eigenvectors.

---

## 4. Principal Components

The **Principal Components** of a dataset are the **eigenvectors** of its covariance matrix.

* **First Principal Component (PC1):** The eigenvector corresponding to the **largest eigenvalue**. This direction captures the maximum variability in the data.
* **Subsequent Components:** PC2 is the eigenvector of the second-largest eigenvalue, and so on. These directions are all mutually orthogonal.

---

## 5. The PCA Algorithm (Step-by-Step)

To reduce an $n$-dimensional dataset to $k$ dimensions ($k < n$):

1. **Standardize the Data:** Center the data by subtracting the mean $\mu$ from each feature.
2. **Compute the Covariance Matrix:** Calculate $\Sigma = \frac{1}{n} C^T C$.
3. **Eigendecomposition:** Find the eigenvalues $\lambda_1, \lambda_2, \dots, \lambda_n$ and their corresponding eigenvectors $\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n$.
4. **Select Top $k$ Components:** Sort eigenvalues in descending order and choose the top $k$ eigenvectors.
5. **Projection:** Project the original $n$-dimensional data onto the space spanned by these $k$ eigenvectors. The resulting data is now $k$-dimensional.

---

## 6. Significance and Applications

* **Information Preservation:** Even though we reduce dimensions, we keep the components with the highest eigenvalues, which represent the "signal," while discarding those with low eigenvalues, which often represent "noise."
* **Data Visualization:** Reducing complex data to 2D or 3D for plotting.
* **Noise Reduction:** By eliminating low-variance components, we can clean datasets.
* **Connection to SVD:** PCA is mathematically equivalent to performing Singular Value Decomposition on the centered data matrix.

---
