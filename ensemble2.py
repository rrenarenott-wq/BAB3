# Example 3.25 Ensemble Learning (Diabetes Regression)
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# 1. Load dataset Diabetes
X, y = load_diabetes(return_X_y=True)

# 2. Inisialisasi Regressor
reg1 = GradientBoostingRegressor(random_state=1)
reg2 = RandomForestRegressor(random_state=1)
reg3 = LinearRegression()
reg4 = MLPRegressor(max_iter=1000, random_state=1)  # Tambah max_iter agar konvergen

# 3. Latih masing-masing regressor
reg1.fit(X, y)
reg2.fit(X, y)
reg3.fit(X, y)
reg4.fit(X, y)

# 4. Inisialisasi dan latih VotingRegressor
ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3), ('NN', reg4)])
ereg.fit(X, y)

# 5. Prediksi 20 sampel pertama
Xt = X[:20]
pred1 = reg1.predict(Xt)
pred2 = reg2.predict(Xt)
pred3 = reg3.predict(Xt)
pred4 = reg4.predict(Xt)
pred5 = ereg.predict(Xt)

# 6. Visualisasi Hasil Prediksi
plt.figure(figsize=(10, 6))
plt.plot(pred1, 'gd', label='GradientBoostingRegressor')
plt.plot(pred2, 'b^', label='RandomForestRegressor')
plt.plot(pred3, 'ys', label='LinearRegression')
plt.plot(pred4, 'kx', label='MLPRegressor')
plt.plot(pred5, 'r*', ms=10, label='VotingRegressor')

plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.ylabel('predicted')
plt.xlabel('training samples')
plt.legend(loc="best")
plt.title('Regressor predictions and their average')
plt.show()