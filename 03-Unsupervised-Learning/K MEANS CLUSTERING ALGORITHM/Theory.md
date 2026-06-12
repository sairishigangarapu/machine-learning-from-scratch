# K-Means Clustering & Unsupervised Learning

## 1. Supervised vs. Unsupervised Learning

### Supervised Learning
* **Data:** Labeled (Input $X$ + Output $y$).
* **Goal:** Map inputs to known outputs (Prediction).
* **Examples:** Regression (House Prices), Classification (Spam Filter).

### Unsupervised Learning
* **Data:** Unlabeled (Input $X$ only).
* **Goal:** Discover hidden structures, patterns, or groupings.
* **Examples:** Clustering (Customer Segmentation), Dimensionality Reduction (PCA).

| Supervised | Unsupervised |
| :--- | :--- |
| Labels available | No labels |
| Classification, Regression | Clustering, Dimensionality Reduction |

---

## 2. K-Means Clustering Algorithm

K-Means is an iterative algorithm that partitions a dataset into $K$ distinct, non-overlapping subgroups (clusters) where each data point belongs to the cluster with the nearest mean.

### Key Concept: The Centroid
A **Centroid** is the arithmetic mean position of all the points in a specific cluster. It acts as the "center of gravity" for that group.

### The Algorithm (Lloyd’s Algorithm)
1. **Initialization:** Randomly select $K$ points as initial centroids.
2. **Assignment:** Assign each data point to the nearest centroid (typically using Euclidean Distance).
3. **Update:** Recalculate the centroids by taking the mean of all points assigned to that cluster.
4. **Repeat:** Steps 2 & 3 until the centroids stop moving (Convergence).

---

## 3. Choosing $K$: The Elbow Method

One of the biggest challenges in K-Means is deciding how many clusters ($K$) to use. We use the **Elbow Method** to find the sweet spot.

### The Metric: Sum of Squared Errors (SSE)
Also known as "Inertia," this measures how far the data points are from their assigned centroids.

$$
\text{SSE} = \sum_{i=1}^{K} \sum_{x \in C_i} || x - \mu_i ||^2
$$

| Term | Definition | Significance |
| :--- | :--- | :--- |
| $\text{SSE}$ | Sum of Squared Errors (Inertia) | Total within-cluster variance — measures how "tight" the clusters are |
| $K$ | Number of clusters | The hyperparameter we're trying to choose |
| $C_i$ | The set of points assigned to cluster $i$ | All data points closest to centroid $\mu_i$ |
| $x$ | A single data point | One observation in the dataset |
| $\mu_i$ | Centroid (mean) of cluster $i$ | The center of mass of all points in cluster $i$ |
| $\|x - \mu_i\|^2$ | Squared Euclidean distance from point to its centroid | How far the point is from its assigned center — the "cost" of assigning it there |
| $\sum_{x \in C_i}$ | Sum over all points in cluster $i$ | Total cost for cluster $i$ |

### Interpretation
1. Run K-Means for a range of $K$ (e.g., 1 to 10).
2. Plot **$K$ vs. SSE**.
3. As $K$ increases, SSE decreases (distortions get smaller).
4. The **"Elbow"** is the point where the rate of decrease sharply slows down. This indicates the optimal tradeoff between error and model complexity.

---
**External Exercise:** [Codebasics K-Means Lab](https://github.com/codebasics/py/blob/master/ML/13_kmeans/13_kmeans_tutorial.ipynb)
