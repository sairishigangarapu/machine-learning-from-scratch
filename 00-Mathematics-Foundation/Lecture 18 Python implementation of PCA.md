## PCA: Python Implementation (From Scratch & Scikit-Learn)

*Essential Mathematics for ML — Applied Structured Notes*

---

## 1. Introduction to Dimensionality Reduction in Practice

### The "Curse of Dimensionality"
Regardless of your hardware, dealing with datasets containing thousands of redundant features will inevitably crash or massively slow down your Machine Learning pipeline. This phenomenon is known as the **Curse of Dimensionality**.

The practical solution is **Dimensionality Reduction**. Having covered the rigorous mathematics of Principal Component Analysis (PCA) in the previous lectures, we will now implement it entirely in Python. We will do this two ways:

1. **From Scratch (The Hacker's Way):** Using pure NumPy to build the covariance matrix and extract eigenvalues, proving we understand the engine.
2. **The Production Way:** Using `scikit-learn` for a fast, optimized 5-line implementation.

---

## 2. Implementation 1: PCA strictly "From Scratch" (NumPy)

Let's generate a simple 2-dimensional synthetic dataset (11 samples, 2 features) and manually reduce it to 1 dimension.

### Step 1: Create and Visualize the Data
```python
import numpy as np
import matplotlib.pyplot as plt

# Row 1: Feature X1, Row 2: Feature X2
row1 = [1, 3, 5, 7, 9, 13, 18, 20, 21, 24, 26]
row2 = [5, 7, 11, 14, 15, 17, 18, 19, 21, 24, 26]

# Stack arrays into a 2x11 matrix, then transpose to an 11x2 Data Matrix (Samples x Features)
data = np.array([row1, row2]).T

# Visualize the original distribution
plt.scatter(data[:, 0], data[:, 1], color='orange')
plt.title("Original 2D Data")
plt.show()
```

### Step 2: Zero-Centering (Crucial Prerequisite)
Before PCA can be applied, the data *must* have a mean of zero. If we don't shift the data to the origin $(0,0)$, PCA will attempt to draw a component pointing from the origin to the data blob, completely ruining the variance analysis.

```python
# Subtract the mean of each column from the respective column
X_meaned = data - np.mean(data, axis=0)

plt.scatter(X_meaned[:, 0], X_meaned[:, 1], color='blue')
plt.title("Zero-Centered Data")
plt.show()
```

### Step 3: Compute the Covariance Matrix
The Covariance Matrix captures exactly how Feature 1 and Feature 2 vary together. Since we have 2 features, the covariance matrix must be a $2 \times 2$ symmetric matrix.

```python
# rowvar=False expects data as (Samples x Features)
cov_matrix = np.cov(X_meaned, rowvar=False)
print("Covariance Matrix:\n", cov_matrix)
```

### Step 4: Eigendecomposition
We extract the Eigenvalues (the amount of variance) and the Eigenvectors (the directions of that variance).

```python
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("Eigenvalues:", eigenvalues) 
# Evaluates to roughly: [119.29, 2.26] -> The first is massively larger!
```

### Step 5: Sorting and Dimensionality Reduction
The largest eigenvalue ($\approx 119.29$) corresponds to the **Principal Component Direction**. The data varies vastly more along this vector than the other.

To reduce our 2D data to 1D, we project our centered data strictly onto this single, dominant eigenvector.

```python
# 1. Sort the indices of the eigenvalues in descending order
sorted_index = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[:, sorted_index]

# 2. Select the top N components (in this case, just N=1)
n_components = 1
principal_vector = sorted_eigenvectors[:, 0:n_components]

# 3. Project the data! (11x2 matrix multiplied by a 2x1 vector = 11x1 reduced data)
X_reduced = np.dot(X_meaned, principal_vector)

print("Reduced 1D Data Shape:", X_reduced.shape)
```

---

## 3. Implementation 2: PCA using Scikit-Learn

While building it from scratch teaches you the math, production systems rely on `scikit-learn`. `sklearn` uses a highly optimized Singular Value Decomposition (SVD) solver under the hood, mathematically bypassing the need to explicitly construct a massive Covariance Matrix.

```python
from sklearn.decomposition import PCA

# 1. Initialize the PCA object, specifying the desired number of final dimensions
pca = PCA(n_components=1)

# 2. Fit and Transform the centered data in one line
X_reduced_sklearn = pca.fit_transform(X_meaned)

print("Reduced 1D Data:", X_reduced_sklearn)
```
*(Notice how a 20-line mathematical derivation shrinks cleanly into 3 lines of production code!)*

---

## 4. Real-World Example: The Iris Dataset (Reducing 4D to 2D)

Let's scale up. The famous **Iris Dataset** contains 150 flower samples. Each sample has **4 features**: 

1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

Our data matrix size is $150 \times 4$. Humans cannot visually plot 4-dimensional data. Our objective is to compress this down to $2$ dimensions so we can visualize it on a standard scatter plot, while retaining maximum class separability.

### The Code Implementation

```python
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Load the 4D Dataset
iris = datasets.load_iris()
X = iris.data    # Shape: (150, 4)
y = iris.target  # Targets: Setosa (0), Versicolor (1), Virginica (2)

# 2. Apply PCA tightly into 2 Dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X) # New Shape: (150, 2)

# 3. Visualize the mathematically compressed data
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis')
plt.title("4D Iris Dataset Compressed to 2D using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, ticks=[0, 1, 2], label='Iris Species')
plt.show()
```

### What happened mathematically?

1. `scikit-learn` secretly calculated a $4 \times 4$ covariance matrix.
2. It found $4$ eigenvalues ($\lambda_1, \lambda_2, \lambda_3, \lambda_4$).
3. It selected the $2$ largest eigenvalues, isolated their 2 corresponding eigenvectors, and multiplied the $150 \times 4$ data matrix by a $4 \times 2$ projection matrix.
4. The output is a $150 \times 2$ matrix. 

**The Visual Result:** When plotted, you will clearly see three distinct clusters of data in 2D space. Even though we literally deleted 50% of the dataset's structural dimensions (dropping from 4 to 2), the principal components mathematically salvaged enough variance that a linear classifier could easily draw a hyper-plane between the flowers!

---

## 5. Summary and Next Steps

We have proven that PCA is not just abstract theory:

* We manually extracted Python eigenvalues to construct the principal component projection line.
* We utilized `scikit-learn` to dramatically compress a 4D database into a 2D visualization capable of separating flower species almost perfectly.

However, PCA is utterly "blind" to class labels. It maximizes *variance*, not *classification separation*. What if we want a dimensionality reduction algorithm that is explicitly designed specifically to push different classes apart? In the next module, we explore **Linear Discriminant Analysis (LDA)**.

---

### Applied Practice 🚀
Now that you have mastered the foundational mathematics of PCA, see it applied to a real-world **Image Compression & Logistic Regression Pipeline** (The MNIST Digits Dataset) by running the lab in our specific Unsupervised Learning module:
* **[Unsupervised Learning - PCA Lab](../../03-Unsupervised-Learning/PRINCIPAL%20COMPONENT%20ANALYSIS/pca_lab.py)**
* **[Unsupervised Learning - PCA Theory](../../03-Unsupervised-Learning/PRINCIPAL%20COMPONENT%20ANALYSIS/Theory.md)**

