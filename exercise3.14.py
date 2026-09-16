# Exercise 3.14 K-means with 2 Additional Points per Group
from sklearn.cluster import KMeans
import numpy as np

# Data asli + 2 titik baru per kelompok (total 10 titik data)
X = np.array([
    # Kelompok 1
    [1, 2, 3], [1, 4, 2], [1, 0, 3], [2, 2, 2], [1, 3, 3],
    # Kelompok 2
    [10, 2, 4], [9, 4, 3], [11, 0, 2], [10, 3, 3], [9, 1, 4]
])

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

print("Cluster Labels:")
print(kmeans.labels_)
print("\nCluster Centers:")
print(kmeans.cluster_centers_)