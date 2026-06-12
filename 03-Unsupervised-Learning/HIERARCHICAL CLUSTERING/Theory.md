# Hierarchical Clustering 🌳

## 1. Concept Overview
**Hierarchical Clustering** builds a tree-like hierarchy of clusters (a **dendrogram**), either by progressively merging small clusters (**agglomerative**) or splitting large clusters (**divisive**).

Unlike K-Means, you do **not** need to specify $k$ in advance — you can "cut" the dendrogram at any level.

---

## 2. Two Approaches

### A. Agglomerative (Bottom-Up)
1. Start with each point as its own cluster.
2. Find the two closest clusters, merge them.
3. Update the distance matrix.
4. Repeat until only one cluster remains.

### B. Divisive (Top-Down)
1. Start with all points in one cluster.
2. Split the cluster into two.
3. Keep splitting until each point is its own cluster.

> **Agglomerative is far more common** in practice. Most libraries (including sklearn) implement only agglomerative.

---

## 3. Linkage Criteria: How to Measure Distance Between Clusters

| Linkage | Distance Between Clusters | Character |
| :--- | :--- | :--- |
| **Single** | Minimum distance between any two points in different clusters | Can form "chain-like" clusters |
| **Complete** | Maximum distance between any two points in different clusters | Compact, spherical clusters |
| **Average** | Average distance between all pairs | Compromise between single and complete |
| **Ward** | Minimizes total within-cluster variance | Tends to produce equally-sized clusters (default) |

---

## 4. The Dendrogram

A dendrogram is a tree diagram showing the sequence of merges and the distances at which they occurred. The **height** of each merge represents the distance between the two clusters being combined.

### How to Read It
* **Y-axis:** Distance (or dissimilarity) at which merges happen.
* **Cutting horizontally** at a chosen height gives you the desired number of clusters.
* **Long gaps** between merge heights indicate natural cluster boundaries.

---

## 5. Code Example

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

X, _ = make_blobs(n_samples=150, centers=3, random_state=42)

# Dendrogram
linkage_matrix = linkage(X, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, truncate_mode='lastp', p=30)
plt.title('Ward Linkage Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
plt.axhline(y=15, color='r', linestyle='--', label='Cut threshold')
plt.legend()
plt.show()

# Fit Agglomerative Clustering
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = hc.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
plt.title('Agglomerative Clustering (Ward, k=3)')
plt.show()
```

---

## 6. Agglomerative vs. K-Means

| Feature | Agglomerative | K-Means |
| :--- | :--- | :--- |
| Number of clusters | Choose by cutting dendrogram | Must specify $k$ upfront |
| Cluster shape | Arbitrary (any linkage) | Spherical (assumes convex) |
| Scalability | $O(n^3)$ memory, $O(n^2 \log n)$ time | $O(nkt)$ — much faster |
| Determinism | Deterministic | Random initialization |
| Best for | Small datasets, exploring structure | Large datasets, known $k$ |

---

## 7. Advantages & Disadvantages

### ✅ Pros
* No need to pre-specify $k$ — the dendrogram reveals structure.
* Can produce arbitrarily shaped clusters (with single/average linkage).
* Deterministic and reproducible.

### ❌ Cons
* **Computationally expensive** — $O(n^2)$ distance matrix makes it impractical for >10K points.
* **Sensitive to noise** — single linkage is prone to chaining.
* **Irreversible merges** — once two clusters are merged, they cannot be split.

---

**Previous:** [DBSCAN](../DBSCAN/Theory.md) | **Next:** [PCA](../PRINCIPAL%20COMPONENT%20ANALYSIS/Theory.md) | **Related:** [K-Means](../K%20MEANS%20CLUSTERING%20ALGORITHM/Theory.md)
