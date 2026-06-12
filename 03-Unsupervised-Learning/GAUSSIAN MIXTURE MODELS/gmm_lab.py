import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score

def run_gmm_lab():
    # ---------------------------------------------------------
    # 1. Generate Data with Elliptical Clusters
    # ---------------------------------------------------------
    np.random.seed(42)
    # Cluster 1: tight, elongated
    c1 = np.random.randn(100, 2) @ np.array([[1.5, 0.5], [0.5, 0.3]]) + np.array([0, 0])
    # Cluster 2: circular
    c2 = np.random.randn(100, 2) * 0.8 + np.array([4, 3])
    # Cluster 3: wide, tilted
    c3 = np.random.randn(100, 2) @ np.array([[0.4, 0.8], [0.8, 1.5]]) + np.array([-2, 4])
    X = np.vstack([c1, c2, c3])

    # ---------------------------------------------------------
    # 2. BIC for Model Selection
    # ---------------------------------------------------------
    bic_scores = []
    aic_scores = []
    k_range = range(1, 7)

    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        aic_scores.append(gmm.aic(X))

    best_k = list(k_range)[np.argmin(bic_scores)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(k_range), bic_scores, marker='o', label='BIC')
    ax1.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
    ax1.set_xlabel('Number of Components (k)')
    ax1.set_ylabel('BIC')
    ax1.set_title('BIC for Model Selection')
    ax1.legend()
    ax1.grid()

    ax2.plot(list(k_range), aic_scores, marker='s', color='green', label='AIC')
    ax2.set_xlabel('Number of Components (k)')
    ax2.set_ylabel('AIC')
    ax2.set_title('AIC for Model Selection')
    ax2.legend()
    ax2.grid()
    plt.tight_layout()
    plt.show()
    print(f"✅ Best k by BIC: {best_k}")

    # ---------------------------------------------------------
    # 3. Fit GMM (k=3)
    # ---------------------------------------------------------
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42, n_init=5)
    labels_gmm = gmm.fit_predict(X)
    probs = gmm.predict_proba(X)

    print(f"\n📊 GMM Parameters:")
    print(f"  Mixing weights: {gmm.weights_}")
    for i in range(3):
        print(f"  Cluster {i}: mean={gmm.means_[i].round(2)}, "
              f"cov_diag={np.diag(gmm.covariances_[i]).round(2)}")
    print(f"  Silhouette: {silhouette_score(X, labels_gmm):.3f}")

    # ---------------------------------------------------------
    # 4. Compare: K-Means vs GMM
    # ---------------------------------------------------------
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_km = km.fit_predict(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(X[:, 0], X[:, 1], c=labels_km, cmap='viridis', s=30)
    axes[0].set_title(f'K-Means (Silhouette: {silhouette_score(X, labels_km):.3f})')

    axes[1].scatter(X[:, 0], X[:, 1], c=labels_gmm, cmap='viridis', s=30)
    axes[1].set_title(f'GMM (Silhouette: {silhouette_score(X, labels_gmm):.3f})')

    for ax in axes:
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 5. Soft Assignment Visualization
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
        sc = axes[i].scatter(X[:, 0], X[:, 1], c=probs[:, i], cmap='Reds', s=30)
        axes[i].set_title(f'P(Component {i})')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
        plt.colorbar(sc, ax=axes[i])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_gmm_lab()
