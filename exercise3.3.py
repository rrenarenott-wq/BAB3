from sklearn import svm
import pandas as pd
from matplotlib import pyplot

# 1. Load dataset Iris dari URL (Example 3.4)
df = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')

# 2. Ambil 2 fitur pertama (Sepal Length & Sepal Width)
X = df.iloc[:, :2].values
s = df['species']

# Encode label species ke angka (0, 1, 2) untuk pewarnaan plot
d = dict([(y, x) for x, y in enumerate(sorted(set(s)))])
y = [d[x] for x in s]

# 3. Buat Scatter Plot dari Sepal Length vs Sepal Width (Exercise 3.3)
pyplot.figure(figsize=(8, 6))
scatter = pyplot.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
pyplot.xlabel('Sepal Length (cm)')
pyplot.ylabel('Sepal Width (cm)')
pyplot.title('Exercise 3.3: Scatter Plot of Sepal Features')
pyplot.colorbar(scatter, ticks=[0, 1, 2], label='Species')
pyplot.show()

# 4. Training SVM dan Prediksi
clf = svm.SVC()
clf.fit(X, y)

p = clf.predict([[5.4, 3.2]])
print("Hasil Prediksi:", p)