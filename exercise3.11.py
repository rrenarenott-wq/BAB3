import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Menambah titik data x dan y (misal 50 data point dengan sedikit noise)
np.random.seed(42)
x = np.linspace(0, 10, 50).reshape(-1, 1)
y = 2.5 * x.squeeze() + 5 + np.random.normal(0, 2, size=50)

# 2. Fit model Linear Regression
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

# 3. Visualisasi plot sesuai instruksi Exercise 3.11
plt.figure(figsize=(8, 6))

# Scatter plot untuk data points
plt.scatter(x, y, color='blue', label='Data Points')

# Garis regresi linear
plt.plot(x, y_pred, color='red', linewidth=2, label='Regression Line')

# Menambahkan komponen grafik yang diminta
plt.xlabel('Feature (X)')
plt.ylabel('Target (Y)')
plt.title('Exercise 3.11: Linear Regression Plot')
plt.legend()
plt.grid(True)

# Tampilkan plot
plt.show()