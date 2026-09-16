# Exercise 3.15 K-means with 3rd Group of Points
from sklearn.cluster import KMeans
import numpy as np

# Data 3 Kelompok
X = np.array([
    # Kelompok 1
    [1, 2, 3], [1, 4, 2], [1, 0, 3],
    # Kelompok 2
    [10, 2, 4], [9, 4, 3], [11, 0, 2],
    # Kelompok 3 (baru)
    [20, 20, 20], [21, 19, 20], [19, 21, 22]
])

# Memakai 3 cluster
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

print("Cluster Labels:")
print(kmeans.labels_)
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Contoh prediksi 1 sampel baru
sample_prediction = kmeans.predict([[20, 20, 21]])
print("\nPrediction for sample [20, 20, 21]:", sample_prediction)