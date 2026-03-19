## LDA: Python Implementation (Scikit-Learn)

*Essential Mathematics for ML — Applied Structured Notes*

---

## 1. LDA vs. PCA — A Quick Recap

Before diving into code, here is a crisp side-by-side comparison. Think of this as your **"Hacker's Decision Tree"**:

| Property | PCA | LDA |
|---|---|---|
| **Type** | Unsupervised (Feature Extraction) | Supervised (Data Classification) |
| **Objective** | Direction of **maximum variance** | Direction of **maximum inter-class separation** |
| **Uses Class Labels?** | No | **Yes** (Fit requires `y`) |
| **Max Output Dimensions** | $\min(n, d)$ | $C - 1$ ($C$ = # of classes) |
| **Preserves Linear Separability?** | Not guaranteed | **Yes**, by design |

> **The Hard Constraint:** For a dataset with $C$ classes, LDA can only project the data into at most $C - 1$ dimensions. 
> * 3 classes? Max 2D.
> * 2 classes? **Max 1D.** 
> This is a fundamental limit of the mathematics (the rank of the between-class scatter matrix $S_B$).

The sklearn API for LDA is: `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis`

---

## 2. Example 1 — Toy 2D Dataset (PCA vs. LDA)

### 2.1 Setup & Data
We define a small synthetic 2D, 2-class dataset (14 samples) to visually compare the projections of PCA and LDA side-by-side.

```python
import numpy as np
import matplotlib.pyplot as plt

# Feature matrix: 14 samples, 2 features (X1 and X2)
# Note: Define as two rows (features), then transpose to (samples x features)
X = np.array([
    [0, 1, 2, 3, 4, 5, 1, 2, 3, 3, 5, 6, 7, 8],  # Feature X1
    [1, 2, 3, 3, 5, 6, 7, 8, 1, 2, 3, 3, 5, 6]   # Feature X2
]).T  # Shape: (14, 2)

# Class labels: first 6 samples → Class 0, last 8 → Class 1
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

# Visualize original 2D data (coloured by class)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title("Original 2D Data (coloured by class)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
```

The data is **linearly separable** in 2D. The challenge is to project it to 1D while *preserving* that separability.

---

### 2.2 Apply PCA (Baseline — no class awareness)

```python
from sklearn.decomposition import PCA

# Reduce to 1 principal component
pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)   # Shape: (14, 1)

# Plot the 1D projection along a horizontal line
plt.scatter(X_pca, np.zeros_like(X_pca), c=y, cmap='viridis')
plt.title("PCA Projection to 1D")
plt.xlabel("PC1")
plt.yticks([])
plt.show()
```

**Observation:** The two classes are now **mixed** in 1D. PCA found the axis of maximum spread, but that axis is *not* aligned with the class boundary. A linear threshold cannot separate them here.

---

### 2.3 Apply LDA (Class-Aware)

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# For 2 classes, LDA can reduce to at most 1D (C - 1 = 1)
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X, y)  # y (class labels) are REQUIRED here

# Plot the 1D LDA projection
plt.scatter(X_lda, np.zeros_like(X_lda), c=y, cmap='viridis')
plt.title("LDA Projection to 1D")
plt.xlabel("LD1")
plt.yticks([])
plt.show()

print("LDA 1D Representation:\n", X_lda)
print("PCA 1D Representation:\n", X_pca)
```

**Observation:** The LDA projection shows two **distinct, non-overlapping clusters** — one for each class. Any threshold point separates them perfectly. This directly contrasts with the PCA result above.

> **Critical Difference:** `fit_transform(X, y)` — LDA's `fit_transform` takes **both** `X` (features) and `y` (class labels). PCA's `fit_transform` takes only `X`.

---

## 3. Engineering Reality: The Moore-Penrose Pseudoinverse

In Lecture 19, we saw the formula $\mathbf{v} = S_W^{-1} (\mu_1 - \mu_2)$. But in the real world, $S_W$ is often **singular** (non-invertible) if our data is collinear or if we have more features than samples (the $d > n$ problem).

### The Fix
Instead of a standard inverse, robust implementations use the **Moore-Penrose Pseudoinverse** ($S_W^{+}$), calculated via SVD. Sklearn handles this internally, but when coding from scratch, you should use `np.linalg.pinv(Sw)` instead of `np.linalg.inv(Sw)`.

> **Hacker's Pro-Tip:** If your LDA starts throwing "Singular Matrix" errors, it's a sign your features are highly redundant. Try running PCA *first* to reduce noise, then pipe that output into LDA.

---

## 3. Example 2 — Wine Dataset (13D → 2D)

### 3.1 Dataset Overview

The **Wine Dataset** is a classic multi-class benchmark included in `scikit-learn`:

| Property | Value |
|---|---|
| Samples | 178 |
| Features | 13 (e.g., Alcohol, Malic Acid, Proline, ...) |
| Classes | 3 (three cultivars of Italian wine) |
| Feature Type | Continuous / Real-valued |
| Missing Values | None |

Since there are **3 classes**, LDA can reduce the data to at most **2 dimensions** ($C - 1 = 2$). PCA has no such restriction but will still be reduced to 2D for a fair visual comparison.

---

### 3.2 Load and Inspect the Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

# Load the dataset
wine = datasets.load_wine()
X = np.array(wine.data)    # Shape: (178, 13)
y = np.array(wine.target)  # Shape: (178,) — classes 0, 1, 2

# Inspect first 5 rows
print(X[:5, :])
print("Class labels:", y)

# View as a labelled DataFrame
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y
print(df.head())
```

---

### 3.3 PCA on the Wine Dataset (13D → 2D)

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)   # Shape: (178, 2)

plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title("Wine Dataset: PCA (13D → 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label='Wine Class')
plt.show()
```

**Observation:** The three wine classes are **mixed and overlapping** after PCA. The projection axis maximised overall data variance, but the classes are no longer cleanly separable in 2D.

---

### 3.4 LDA on the Wine Dataset (13D → 2D)

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 3 classes → maximum 2 LDA components (C - 1 = 2)
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X, y)  # Shape: (178, 2)

plt.figure(figsize=(8, 5))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis')
plt.title("Wine Dataset: LDA (13D → 2D)")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.colorbar(label='Wine Class')
plt.show()
```

**Observation:** The three wine classes form **three clearly separated clusters** in the 2D LDA space. A simple linear classifier can now draw two decision boundaries to perfectly classify the wines — all while operating in just 2 dimensions instead of 13.

---

## 4. LDA as a Classifier (Train/Test Split)

LDA is not just a dimensionality reduction tool — it can *directly classify* unseen data by projecting it onto the LDA subspace and assigning it to the nearest class cluster.

```python
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Split: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train (fit) the LDA model on the training split
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Predict class labels for unseen test samples
y_pred = lda.predict(X_test)

# Evaluate performance
print("True classes:     ", y_test)
print("Predicted classes:", y_pred)
print("Differences:      ", y_test - y_pred)
```

**Result:** On the Wine dataset, LDA typically achieves **~100% accuracy** on the test set. This demonstrates that for linearly separable data, LDA is an extremely powerful and compact classifier — it reduces 13 dimensions to 2, and still perfectly classifies every test sample.

---

## 5. Key API Summary

| Task | Code |
|---|---|
| Import LDA | `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis` |
| Create object | `lda = LinearDiscriminantAnalysis()` |
| Fit + transform (reduction) | `X_lda = lda.fit_transform(X, y)` |
| Fit only (classification) | `lda.fit(X_train, y_train)` |
| Predict classes | `y_pred = lda.predict(X_test)` |
| Max output dimensions | Always $\leq C - 1$ |

> **Important:** Unlike PCA's `fit_transform(X)`, LDA's `fit_transform` always requires **both** `X` and `y` since it is a supervised technique.

---

## 6. Summary and Next Steps

In this lecture we implemented LDA with `scikit-learn` in two settings:

- **Toy Dataset:** Showed visually that PCA mixes classes in 1D, while LDA cleanly separates them.
- **Wine Dataset:** Compressed 13 features to 2 LDA dimensions — maintaining perfect linear separability across 3 wine classes.
- **Classification Pipeline:** Used LDA as a full supervised classifier with a train/test split, achieving ~100% test accuracy on the Wine dataset.

LDA is the go-to method when class information is available and linear separability matters. For unlabelled data or purely exploratory analysis, PCA remains the appropriate tool.

**End of Module:** You are now equipped to handle high-dimensional classification. In the next lecture, we move to the math behind **Linear Regression**—the engine of modern forecasting.
