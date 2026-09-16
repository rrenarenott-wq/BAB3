from sklearn import svm

# X berisi 6 sampel data: [Tinggi (cm), Berat (kg), Ukuran Sepatu (UK)]
X = [
    [170, 70, 10],  # Sampel 1 (Pria)
    [180, 80, 12],  # Sampel 2 (Pria)
    [175, 75, 11],  # Sampel 3 (Pria) - Tambahan
    [170, 65, 8],   # Sampel 4 (Wanita)
    [160, 55, 7],   # Sampel 5 (Wanita)
    [155, 50, 6]    # Sampel 6 (Wanita) - Tambahan
]

# y berisi 6 label kelas (0: Male, 1: Female)
y = [0, 0, 0, 1, 1, 1]

# Inisialisasi dan training model SVM
clf = svm.SVC()
clf.fit(X, y)

# Prediksi sampel baru [160 cm, 60 kg, ukuran 7]
p = clf.predict([[160, 60, 7]])
print(p)