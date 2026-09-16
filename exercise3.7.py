import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

# 1. Load Breast Cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# 2. Inisialisasi dan jalankan PCA (reduksi menjadi 2 komponen utama)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 3. Visualisasi data hasil PCA (Exercise 3.7)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", edgecolors="k"
)
plt.xlabel("First Principal Component (PCA1)")
plt.ylabel("Second Principal Component (PCA2)")
plt.title("Exercise 3.7: PCA on Breast Cancer Dataset")
plt.colorbar(
    scatter, ticks=[0, 1], label="Target (0: Malignant, 1: Benign)"
)

# Tampilkan plot agar tidak langsung tertutup
plt.show()

# 4. Evaluasi varians yang dijelaskan oleh 2 komponen
print(
    "Explained variance ratio (PCA1 & PCA2):", pca.explained_variance_ratio_
)
print(
    f"Total varians yang dijelaskan: {sum(pca.explained_variance_ratio_) * 100:.2f}%"
)