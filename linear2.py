# Example 3.15a Linear Regression
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# 1. Data masukan
x = [0, 1, 2, 3, 4]
y = [3, 5, 5, 6, 7]

# 2. Menambahkan konstanta/intercept (x1) untuk statsmodels
x1 = sm.add_constant(x)

# 3. Fit model Ordinary Least Squares (OLS)
model = sm.OLS(y, x1)
results = model.fit()

# 4. Menampilkan parameter (slope & intercept) dan tabel hasil statistik
print(results.params)
print(results.summary())

# 5. Prediksi dan visualisasi plot
y_pred = results.predict(x1)

plt.scatter(x, y)
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x, y_pred, "r")
plt.show()