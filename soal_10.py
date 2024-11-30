import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Dataset baru
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
Y = np.array([3, 7, 13, 21, 31, 43, 57, 73, 91, 111]).reshape(-1, 1)

# Membagi dataset menjadi 80% data latih dan 20% data uji
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Membuat model regresi linear
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)

# Membuat model regresi polinomial derajat 2
poly_features_2 = PolynomialFeatures(degree=2)
X_train_poly_2 = poly_features_2.fit_transform(X_train)

poly_model_2 = LinearRegression()
poly_model_2.fit(X_train_poly_2, Y_train)

# Membuat prediksi untuk seluruh dataset
X_sorted = np.sort(X, axis=0)  # Urutkan X untuk membuat plot mulus
Y_pred_linear_all = linear_model.predict(X_sorted)
Y_pred_poly_2_all = poly_model_2.predict(poly_features_2.transform(X_sorted))

# Evaluasi model
mse_linear = mean_squared_error(Y_test, linear_model.predict(X_test))
mse_poly_2 = mean_squared_error(Y_test, poly_model_2.predict(poly_features_2.transform(X_test)))

print(f"Mean Squared Error (Linear): {mse_linear:.2f}")
print(f"Mean Squared Error (Polinomial Degree 2): {mse_poly_2:.2f}")

# Plot hasil regresi untuk seluruh dataset
plt.figure(figsize=(10, 6))
plt.scatter(X_train, Y_train, color='blue', label='Data Latih')  # Data latih
plt.scatter(X_test, Y_test, color='orange', label='Data Uji')  # Data uji
plt.plot(X_sorted, Y_pred_linear_all, color='red', label='Regresi Linear')  # Garis regresi linear
plt.plot(X_sorted, Y_pred_poly_2_all, color='green', label='Regresi Polinomial Degree 2')  # Garis regresi polinomial degree 2
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regresi Linear dan Polinomial Degree 2 (Data Latih & Uji)')
plt.legend()
plt.show()
