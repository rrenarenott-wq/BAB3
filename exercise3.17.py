# Exercise 3.17 Ensemble Learning with Added KNN Classifier
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 1. Load dataset Iris
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

# 2. Inisialisasi masing-masing pengklasifikasi (ditambah KNN)
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
clf4 = SVC()
clf5 = KNeighborsClassifier(n_neighbors=5)  # Classifier tambahan: KNN

# 3. Gabungkan semua pengklasifikasi ke dalam VotingClassifier
eclf = VotingClassifier(
    estimators=[
        ('lr', clf1), 
        ('rf', clf2), 
        ('gnb', clf3), 
        ('svc', clf4), 
        ('knn', clf5)
    ],
    voting='hard'
)

# 4. Evaluasi performa masing-masing classifier dan model ensemble
labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'SVM', 'KNN', 'Ensemble']
classifiers = [clf1, clf2, clf3, clf4, clf5, eclf]

for clf, label in zip(classifiers, labels):
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))