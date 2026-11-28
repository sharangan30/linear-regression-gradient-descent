import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def MSE(length, error):
    return ((1/length * error * error).sum())


def gradient_descent(X, y, learning_rate = 0.005, max_steps = 50000, tolerance = 0.00001):
    b = 0
    losses = []
    n = X.shape[0]
    k = X.shape[1]
    w = np.zeros(k)

    for step in range(max_steps):
        errors = X @ w + b - y
        meanSquaredError = MSE (n, errors)
        losses.append(meanSquaredError)
        grad_w = 2/n * X.T @ errors
        grad_b = 2/n * errors.sum()

        if (np.abs(grad_w).sum() + abs(grad_b) < tolerance):
            break
        w = w - learning_rate*grad_w
        b = b - learning_rate*grad_b
    return w, b, losses


x1 = [1, 2, 3, 4, 5, 6]
x2 = [2, 1, 4, 3, 6, 5]
x3 = [5, 3, 6, 2, 4, 1]
y  = [8, 9, 17, 20, 29, 31]

X = np.array([x1, x2, x3]).T
y = np.array(y)

w, b, losses = gradient_descent(X, y, 0.02)

w_gd = w
b_gd = b

model = LinearRegression()
model.fit(X, y)
print(model.coef_, model.intercept_)

print(np.abs(w_gd - model.coef_), abs(b_gd - model.intercept_))

plt.figure(figsize=(10, 6))
rates = [0.005, 0.02, 0.01, 0.05]
for lr in rates:
    w, b, losses = gradient_descent(X, y, lr)
    plt.plot(losses, label = lr)
plt.xlabel("Step")
plt.ylabel("MSE")
plt.title("Loss over training")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.ylim(1e-10, 1e3)
plt.show()