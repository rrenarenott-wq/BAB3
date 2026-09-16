from sklearn import datasets
from sklearn import svm

# 1. Load Iris Dataset
iris = datasets.load_iris()

# 2. Menggunakan fitur ke-3 dan ke-4 (Petal Length dan Petal Width)
X = iris.data[:, 2:4]  # Indeks 2 dan 3
y = iris.target

# 3. Inisialisasi dan Training Model SVM
clf = svm.SVC()
clf.fit(X, y)

# 4. Evaluasi Akurasi
accuracy = clf.score(X, y)
print(f"Akurasi menggunakan Petal (Features 3 & 4): {accuracy * 100:.2f}%")