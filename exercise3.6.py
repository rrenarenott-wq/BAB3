from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Generate dataset buatan: 2,000 sampel & 6 fitur (Exercise 3.6)
X, y = make_classification(
    n_samples=2000, 
    n_features=6, 
    n_informative=4, 
    n_redundant=2, 
    n_classes=2, 
    random_state=42
)

# 2. Split data: 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Fitur Reduksi Dimensi & Klasifikasi dengan LDA
lda = LDA(n_components=1)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# 4. Evaluasi Akurasi Prediksi
y_pred = lda.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Bentuk data awal (X): {X.shape}")
print(f"Bentuk data setelah LDA: {X_train_lda.shape}")
print(f"Akurasi Model LDA (2000 sampel, 6 fitur): {accuracy * 100:.2f}%")