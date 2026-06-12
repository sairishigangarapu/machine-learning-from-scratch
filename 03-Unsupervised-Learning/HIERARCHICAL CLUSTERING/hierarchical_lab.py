import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs, make_moons
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score

def run_hierarchical_lab():
 # ---------------------------------------------------------
 # 1. Generate Data
 # ---------------------------------------------------------
 X_blobs, _ = make_blobs(n_samples=150, centers=3, cluster_std=1.0, random_state=42)
 X_moons, _ = make_moons(n_samples=200, noise=0.08, random_state=42)

 # ---------------------------------------------------------
 # 2. Dendrogram (Ward Linkage)
 # ---------------------------------------------------------
 fig, axes = plt.subplots(1, 2, figsize=(14, 5))

 linkage_ward = linkage(X_blobs, method='ward')
 dendrogram(linkage_ward, truncate_mode='lastp', p=30, ax=axes[0])
 axes[0].set_title('Ward Linkage Dendrogram')
 axes[0].set_xlabel('Cluster Size')
 axes[0].set_ylabel('Distance')
 axes[0].axhline(y=15, color='r', linestyle='--', label='Cut → k=3')
 axes[0].legend()

 linkage_avg = linkage(X_blobs, method='average')
 dendrogram(linkage_avg, truncate_mode='lastp', p=30, ax=axes[1])
 axes[1].set_title('Average Linkage Dendrogram')
 axes[1].set_xlabel('Cluster Size')
 axes[1].set_ylabel('Distance')
 plt.tight_layout()
 plt.show()

 # ---------------------------------------------------------
 # 3. Compare Linkage Methods on Blobs
 # ---------------------------------------------------------
 print(" Linkage Comparison (3 clusters on Blobs data):")
 for method in ['ward', 'complete', 'average', 'single']:
 hc = AgglomerativeClustering(n_clusters=3, linkage=method)
 labels = hc.fit_predict(X_blobs)
 sil = silhouette_score(X_blobs, labels)
 print(f" {method:10s} → Silhouette: {sil:.3f}")

 # ---------------------------------------------------------
 # 4. Compare Linkage on Moons (non-spherical)
 # ---------------------------------------------------------
 print("\n Linkage Comparison (2 clusters on Moons data):")
 for method in ['ward', 'complete', 'average', 'single']:
 hc = AgglomerativeClustering(n_clusters=2, linkage=method)
 labels = hc.fit_predict(X_moons)
 sil = silhouette_score(X_moons, labels)
 print(f" {method:10s} → Silhouette: {sil:.3f}")

 # ---------------------------------------------------------
 # 5. Visualize Best Result
 # ---------------------------------------------------------
 hc_best = AgglomerativeClustering(n_clusters=3, linkage='ward')
 labels = hc_best.fit_predict(X_blobs)

 plt.figure(figsize=(7, 5))
 plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels, cmap='viridis', s=40)
 plt.title('Agglomerative Clustering (Ward, k=3)')
 plt.xlabel('Feature 1')
 plt.ylabel('Feature 2')
 plt.grid(True)
 plt.show()

if __name__ == "__main__":
 run_hierarchical_lab()
