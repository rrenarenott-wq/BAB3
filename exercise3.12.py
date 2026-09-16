# Exercise 3.12 Multiple Linear Regression on Linnerud Dataset
from sklearn.datasets import load_linnerud
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Linnerud dataset
linnerud = load_linnerud()
X = linnerud.data
y = linnerud.target

# 2. Split data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Inisialisasi dan latih model Multiple Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Prediksi data testing
y_pred = model.predict(X_test)

# 5. Evaluasi Hasil
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Fitur X (Exercise/Physiological Data):", linnerud.feature_names)
print("Target Y (Physiological Targets):", linnerud.target_names)
print("-" * 50)
print("Coefficients (Bobot Slope):")
print(model.coef_)
print("\nIntercept (Titik Potong):")
print(model.intercept_)
print("-" * 50)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R2 Score: {r2:.4f}")