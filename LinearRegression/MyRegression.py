import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

X = np.array([
    [10, 20, 15],
    [12, 18, 14],
    [14, 17, 16],
    [16, 16, 18],
    [18, 15, 19],
    [20, 14, 20],
    [22, 13, 21],
    [24, 12, 22],
    [26, 11, 23],
    [28, 10, 24]
])
y = np.array([
    50.0,
    54.2,
    53.1,
    58.0,
    63.7,
    61.0,
    67.5,
    66.0,
    70.3,
    73.9
])

class Scaler:
    def __init__(self):
        self.mean = 0
        self.std = 1

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class MyLinearRegression:
    def __init__(self, learning_rate=0.0005, epochs=100000, degree=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.cost_history = []
        self.degree = degree

    def fit(self, X, y, epochs=None, learning_rate=None, degree=None):
        if epochs is None:
            epochs = self.epochs
        if learning_rate is None:
            learning_rate = self.learning_rate
        if degree is not None:
            self.degree = degree

        if self.degree > 1:
            X = self.generate_polynomial_features(X, self.degree)

        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.gradient_descent(X, y, epochs, learning_rate)

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def compute_cost(self, X, y):
        y_pred = self.predict(X)
        m = X.shape[0]
        total_cost = np.sum((y_pred - y)**2) / (2 * m)
        return total_cost

    def compute_gradient(self, X, y, y_pred):
        m = X.shape[0]
        n = X.shape[1]
        dj_dw = np.zeros(n)
        dj_db = 0

        for i in range(m):
            error = y_pred[i] - y[i]
            for j in range(n):
                dj_dw[j] += error * X[i][j]
            dj_db += error

        dj_dw /= m
        dj_db /= m
        return dj_dw, dj_db

    def gradient_descent(self, X, y, iters, learning_rate):
        self.cost_history = []
        for i in range(iters):
            y_pred = self.predict(X)
            dj_dw, dj_db = self.compute_gradient(X, y, y_pred)
            self.w = self.w - learning_rate * dj_dw
            self.b = self.b - learning_rate * dj_db
            if i % math.ceil(iters / 10) == 0:
                cost = self.compute_cost(X, y)
                self.cost_history.append(cost)
                print(f"Epoch {i}: Cost = {cost}")

    def show_cost_graphic(self):
        plt.plot(self.cost_history)
        plt.xlabel("Epochs")
        plt.ylabel("Cost")
        plt.title("Cost and Epoch Relation")
        plt.grid()
        plt.show()

    def evaluate(self, X_test, y_test):
        y_predict = self.predict(X_test)
        m = X_test.shape[0]
        error = y_test - y_predict
        mse = np.sum((error) ** 2) / m
        rmse = np.sqrt(mse)
        mae = np.sum(np.absolute(error)) / m
        r2 = 1 - (np.sum(error ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

        print("\n Mean Squared Error: ", mse)
        print(" Root Mean Squared Error: ", rmse)
        print(" Mean Absolute Error: ", mae)
        print(" R-Squared: ", r2)

        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    def generate_polynomial_features(self, X, degree):
        n_features = X.shape[1]
        new_columns = []
        for d in range(1, degree + 1):
            combs = list(combinations_with_replacement(range(n_features), d))
            for comb in combs:
                new_column = np.prod(X[:, comb], axis=1)
                new_columns.append(new_column.reshape(-1, 1))
        return np.hstack(new_columns)

split_index = int(len(X) * 0.8)

# Veriyi ölçeklendirme (standartlaştırma)
scaler = Scaler()
X_scaled = scaler.transform(X)

X_train = X_scaled[:split_index]
X_test = X_scaled[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

for d in [1, 2]:
    print(f"\n==== DEGREE {d} ====")
    model = MyLinearRegression(degree=d)

    X_train_poly = model.generate_polynomial_features(X_train, d)
    X_test_poly = model.generate_polynomial_features(X_test, d)

    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)

    print("Our predicts:", y_pred)
    print("Real values:", y_test)
    model.evaluate(X_test_poly, y_test)
