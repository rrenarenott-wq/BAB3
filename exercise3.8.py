# Exercise 3.8 Decision Tree Classification on Wine Dataset
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Load Wine dataset dari scikit-learn
X, y = load_wine(return_X_y=True)

# 2. Split data: 50% training dan 50% testing (sesuai Example 3.11)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0
)

# 3. Inisialisasi dan latih model Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# 4. Prediksi dan hitung poin yang terlabeli dengan benar
y_pred = clf.predict(X_test)
N = y_test.shape[0]
C = (y_test != y_pred).sum()

print("Total points: %d Correctly labeled points : %d" % (N, N - C))