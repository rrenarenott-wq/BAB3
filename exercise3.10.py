# Exercise 3.10 Comparison of Classifiers on Diabetes Dataset
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# 1. Load Diabetes dataset
X, y_continuous = load_diabetes(return_X_y=True)

# 2. Konversi target kontinu menjadi 2 kelas klasifikasi (0 = Rendah, 1 = Tinggi)
threshold = np.mean(y_continuous)
y = (y_continuous > threshold).astype(int)

# 3. Split data (50% training, 50% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# 4. Daftar nama classifier
names = [
    "SVM", "Naive Bayes", "LDA", "QDA",
    "Decision Tree", "Random Forest",
    "Nearest Neighbors", "Neural Networks"
]

# 5. Daftar objek classifier (QDA diberi reg_param=0.1 agar tidak LinAlgError)
classifiers = [
    SVC(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(reg_param=0.1),
    DecisionTreeClassifier(random_state=0),
    RandomForestClassifier(random_state=0),
    KNeighborsClassifier(),
    MLPClassifier(alpha=1, max_iter=1000, random_state=0)
]

# 6. Training dan evaluasi akurasi
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f"{name}: {score * 100:.2f}%")