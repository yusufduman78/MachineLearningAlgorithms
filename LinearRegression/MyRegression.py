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

class MyPolynomialFeatures:
    def __init__(self,degree = 1, bias_including = False):
        self.degree = degree
        self.bias_including = bias_including

    def fit(self,X):
        self.n_features = X.shape[1]
        self.combinations_ = []
        for d in range(1, self.degree + 1):
            combs = list(combinations_with_replacement(range(self.n_features), d))
            self.combinations_.extend(combs)


    def transform(self,X):
        new_columns = []
        for comb in self.combinations_:
            new_col = np.prod(X[:,comb], axis = 1)
            new_columns.append(new_col.reshape(-1,1))#reshape(-1,1) -1 -> auto count row 1-> each row one column namely column vector
        result = np.hstack(new_columns)
        if self.bias_including:
            bias_col = np.ones((X.shape[0], 1)) # (X.shape[0], 1) is tuple X.shape is row 1 is column namely X's row counts and one new column
            result = np.hstack((bias_col, result))
        return result

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)


class MyLinearRegression:
    def __init__(self, learning_rate=0.0005, epochs=100000, threshold=1000, method='auto'):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.method = method
        self.cost_history = []

    def fit(self, X, y, epochs=None, learning_rate=None):

        # if method is 'normal' threshold is not taken into account
        if self.method == 'normal' or (self.method == 'auto' and X.shape[0] < self.threshold):
            self.fit_with_normal_equation(X, y)
        else:
            self.fit_with_gradient_descent(X, y, epochs, learning_rate)

    def fit_with_normal_equation(self,X,y):
        if np.allclose(X[:,0], 1):
            self.w = np.linalg.pinv(X.T @ X) @ X.T @ y
            self.b = 0
        else:
            X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
            w_now = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias @ y
            self.b = w_now[0]
            self.w = w_now[1:]
        self.cost_history = [self.compute_cost(X, y)]

    def fit_with_gradient_descent(self,X,y,epochs = None,learning_rate = None):
        if epochs is None:
            epochs = self.epochs
        if learning_rate is None:
            learning_rate = self.learning_rate
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.gradient_descent(X,y,epochs, learning_rate)

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


np.random.seed(42)  # Tekrarlanabilirlik için

n_samples = 50
X = np.random.uniform(10, 30, size=(n_samples, 3))

# Gerçek fonksiyon: y = 3*X1 + 2*X2^2 - X3 + gürültü
y = 3*X[:,0] + 2*(X[:,1]**2) - X[:,2] + np.random.normal(0, 5, size=n_samples)

# Veri bölme
split_index = int(0.8 * n_samples)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Ölçeklendirme
scaler = Scaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

poly = MyPolynomialFeatures(degree=2, bias_including=True)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

model = MyLinearRegression(learning_rate=0.001, epochs=50000)
model.fit(X_train_poly, y_train)

y_pred_train = model.predict(X_train_poly)
y_pred_test = model.predict(X_test_poly)

print("Eğitim seti performansı:")
model.evaluate(X_train_poly, y_train)
print("\nTest seti performansı:")
model.evaluate(X_test_poly, y_test)



