# DBSCAN: Density-Based Clustering

## 1. Concept Overview
**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) groups together points that are closely packed, marking as outliers points that lie alone in low-density regions.

Unlike K-Means, DBSCAN:
* Does **not** require specifying the number of clusters $k$.
* Can find **arbitrarily shaped** clusters (not just spherical).
* Explicitly identifies **noise points** (outliers).

---

## 2. Core Parameters

| Parameter | Definition | Typical Values |
| :--- | :--- | :--- |
| `eps` ($\epsilon$) | Maximum distance between two points for them to be considered neighbors. | Depends on data scale; use k-distance graph to tune. |
| `min_samples` | Minimum number of points required within `eps` to form a dense region (core point). | 2 × (number of features) is a common heuristic. |

---

## 3. Point Types

* **Core Point:** A point with at least `min_samples` neighbors within radius `eps`.
* **Border Point:** Within `eps` of a core point but has fewer than `min_samples` neighbors itself.
* **Noise Point:** Neither core nor border — these are the outliers.

---

## 4. Advantages & Disadvantages

### Pros
* No need to specify $k$ in advance.
* Finds non-linear, arbitrarily shaped clusters.
* Robust to outliers (labels them as noise).
* Works well with spatial data and geospatial datasets.

### Cons
* Struggles with clusters of varying densities.
* Sensitive to `eps` and `min_samples` — poor choices yield bad results.
* High-dimensional data requires careful distance metric selection (curse of dimensionality).

---

## 5. Choosing `eps`: The k-Distance Graph

Plot the distance to the $k$-th nearest neighbor (sorted descending). The "elbow" in this plot is a good estimate for `eps`.

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np, matplotlib.pyplot as plt

nn = NearestNeighbors(n_neighbors=min_samples)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])

plt.plot(k_distances)
plt.xlabel("Points (sorted)")
plt.ylabel(f"{min_samples}-th Nearest Neighbor Distance")
plt.title("k-Distance Graph — choose eps at the elbow")
plt.grid()
plt.show()
```

---

**External Exercise:** [SKLearn DBSCAN Documentation](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
