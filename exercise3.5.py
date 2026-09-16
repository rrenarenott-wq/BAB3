import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# 1. Load dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Split data menjadi training & testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Train model Naive Bayes (GaussianNB)
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 3. Simpan model yang sudah dilatih ke file 'nb_model.pkl'
filename = 'nb_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(gnb, file)
print(f"Model berhasil disimpan ke file '{filename}'.")

# 4. Load kembali model dari file 'nb_model.pkl'
with open(filename, 'rb') as file:
    loaded_model = pickle.dump if False else pickle.load(file)
print(f"Model berhasil dimuat dari file '{filename}'.")

# 5. Melakukan prediksi menggunakan model yang baru dimuat
sample_data = [[5.1, 3.5, 1.4, 0.2]]  # Contoh data sampel
prediction = loaded_model.predict(sample_data)
predicted_class = iris.target_names[prediction[0]]

print(f"Hasil prediksi untuk sampel {sample_data}:")
print(f"Kelas Prediksi: {prediction[0]} ({predicted_class})")