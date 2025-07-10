import math

import numpy as np
import matplotlib.pyplot as plt

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



class MyLinearRegression:
    def __init__(self, learning_rate=0.0005, epochs=100000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.cost_history = []

    def fit(self,X,y, epochs = None, learning_rate = None):
        if epochs is None:
            epochs = self.epochs
        if learning_rate is None:
            learning_rate = self.learning_rate
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.gradient_descent(X, y, epochs, learning_rate)

    def predict(self,X):
        return np.dot(X, self.w) + self.b


    def compute_cost(self,X,y):
        y_pred = self.predict(X)
        m = X.shape[0]
        total_cost = np.sum((y_pred - y)**2) / (2*m)

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

a = MyLinearRegression()
print("Fit started")
a.fit(X,y)
print("finished")

print("our predicts", a.predict(X))

print("real values", y)

new_data = np.array([[30, 9, 25]])
print("Tahmin:", a.predict(new_data))
