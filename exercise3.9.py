# Exercise 3.9 Random Forest on Diabetes Dataset
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Load Diabetes dataset dari scikit-learn
X, y = load_diabetes(return_X_y=True)

# 2. Split data: 50% training dan 50% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0
)

# 3. Inisialisasi dan latih model Random Forest Regressor
model = RandomForestRegressor(random_state=0)
model.fit(X_train, y_train)

# 4. Prediksi data testing
y_pred = model.predict(X_test)

# 5. Hitung nilai evaluasi Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
N = y_test.shape[0]

print("Total test samples: %d" % N)
print("Mean Squared Error (MSE): %.2f" % mse)