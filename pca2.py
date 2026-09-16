import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, decomposition

# 1. Load Iris Dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 2. Plot Data Asli (Sebelum PCA)
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Original Data')
plt.show()  # Menggantikan f.show() agar plot tetap terbuka

# 3. Proses PCA (Reduksi menjadi 2 komponen agar bisa di-plot)
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X1 = pca.transform(X)

# 4. Plot Hasil PCA
plt.figure(2)
plt.scatter(X1[:, 0], X1[:, 1], c=y)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA Data')
plt.show() 