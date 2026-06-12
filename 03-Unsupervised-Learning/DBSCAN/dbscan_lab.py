import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_blobs
from sklearn.metrics import silhouette_score

def run_dbscan_lab():
 # ---------------------------------------------------------
 # 1. Generate Non-Spherical Data (K-Means would fail here)
 # ---------------------------------------------------------
 moons, _ = make_moons(n_samples=300, noise=0.08, random_state=42)
 blobs, _ = make_blobs(n_samples=200, centers=[[2, 2], [5, 5]],
 cluster_std=[0.6, 0.8], random_state=42)
 X = np.vstack([moons, blobs])

 scaler = StandardScaler()
 X_scaled = scaler.fit_transform(X)

 # ---------------------------------------------------------
 # 2. Run DBSCAN
 # ---------------------------------------------------------
 # eps and min_samples chosen via k-distance graph heuristic
 db = DBSCAN(eps=0.3, min_samples=5)
 labels = db.fit_predict(X_scaled)

 n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
 n_noise = list(labels).count(-1)

 print(f" DBSCAN found {n_clusters} clusters and {n_noise} noise points")

 # ---------------------------------------------------------
 # 3. Visualization
 # ---------------------------------------------------------
 core_mask = np.zeros(len(labels), dtype=bool)
 core_mask[db.core_sample_indices_] = True

 unique_labels = set(labels)
 colors = [plt.cm.viridis(i / len(unique_labels)) for i in range(len(unique_labels))]

 plt.figure(figsize=(10, 5))

 # Plot non-noise points
 for label, color in zip(unique_labels, colors):
 if label == -1:
 continue # skip noise for now
 mask = (labels == label)
 plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1],
 c=[color], s=40, label=f'Cluster {label}')

 # Plot noise points
 noise_mask = labels == -1
 plt.scatter(X_scaled[noise_mask, 0], X_scaled[noise_mask, 1],
 c='red', marker='x', s=60, label='Noise')

 plt.title(f'DBSCAN Clustering (eps=0.3, min_samples=5)\n'
 f'{n_clusters} clusters, {n_noise} noise points')
 plt.legend()
 plt.grid(True)
 plt.show()

 # ---------------------------------------------------------
 # 4. Silhouette Score (excluding noise)
 # ---------------------------------------------------------
 non_noise = labels != -1
 if n_clusters >= 2:
 sil = silhouette_score(X_scaled[non_noise], labels[non_noise])
 print(f"Silhouette Score (excl. noise): {sil:.3f}")
 else:
 print("Silhouette Score requires at least 2 non-noise clusters.")

if __name__ == "__main__":
 run_dbscan_lab()
