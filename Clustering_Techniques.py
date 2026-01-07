import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN

# Generate synthetic dataset (moons dataset)
X, y = make_moons(n_samples=500, noise=0.1, random_state=42)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Apply Spectral Clustering
spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
spectral_labels = spectral.fit_predict(X)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Plot results
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
axes[0].set_title("Original Data")

axes[1].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', edgecolors='k')
axes[1].set_title("K-Means Clustering")

axes[2].scatter(X[:, 0], X[:, 1], c=spectral_labels, cmap='viridis', edgecolors='k')
axes[2].set_title("Spectral Clustering")

axes[3].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', edgecolors='k')
axes[3].set_title("DBSCAN Clustering")

plt.show()
