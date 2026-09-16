# Exercise 3.13 K-means Clustering with make_blobs
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 1. Generate sampel data points menggunakan make_blobs
# Membuat 300 sampel data, 3 pusat cluster, dan 2 fitur (2D)
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# 2. Inisialisasi dan latih model K-Means
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

# 3. Prediksi label cluster dan ambil titik pusat (cluster centers)
y_kmeans = kmeans.predict(X)
centers = kmeans.cluster_centers_

# 4. Tampilkan informasi hasil clustering di terminal
print("Cluster Centers (Pusat Cluster):")
print(centers)

# 5. Visualisasi hasil Clustering
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis', label='Data Points')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')

plt.title("Exercise 3.13: K-means Clustering on Synthetic Blobs")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()