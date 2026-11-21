import numpy as np
from sklearn.linear_model import LinearRegression

# x1 = [1,2,3,4,5]
# x2 = [3,5,6,8,11]
# y = [1,4,6,7,10]
# m1 = 0
# m2 = 0
# b = 0
# n = len(x1)
# learning_rate = .015
# step = 0

# # Functions
# def error(x1, x2, y, m1, m2, b):
#     return (m1*x1 + m2*x2 +b-y)
# def MSE(length, errorList):
#     value = 0
#     for i in range(length):
#         value += errorList[i]*errorList[i]
#     return (1/n*value)
# def dervB(errorList, length):
#     return (2/length*(sum(errorList)))
# def dervM(errorList, length, x):
#     value = 0
#     for i in range(length):
#         value += x[i]*errorList[i]
#     return (2/n*value)

# errorList = []
# for i in range(n):
#     errorList.append(error(x1[i], x2[i], y[i], m1, m2, b))
# print(errorList)

# dervWRT_B = dervB(errorList, n)
# print(dervWRT_B)

# dervWRT_M1 = dervM(errorList, n, x1)
# print(dervWRT_M1)

# dervWRT_M2 = dervM(errorList, n, x2)
# print(dervWRT_M2)

# m1 = m1 - learning_rate * dervWRT_M1
# m2 = m2 - learning_rate * dervWRT_M2
# b = b - learning_rate * dervWRT_B
# step += 1
# meanSquaredError = MSE(n, errorList)
# print("step ", step, ": ", meanSquaredError)

# while (((abs(dervWRT_B) + abs(dervWRT_M1) + abs(dervWRT_M2)) > 0.00001) and step <= 5500):

#     errorList = []
#     for i in range(n):
#         errorList.append(error(x1[i], x2[i], y[i], m1, m2, b))
#     dervWRT_B = dervB(errorList, n)
#     dervWRT_M1 = dervM(errorList, n, x1)
#     dervWRT_M2 = dervM(errorList, n, x2)
#     m1 = m1 - learning_rate * dervWRT_M1
#     m2 = m2 - learning_rate * dervWRT_M2
#     b = b - learning_rate * dervWRT_B
#     step += 1
#     meanSquaredError = MSE (n, errorList)
#     print("step ", step, ": ", meanSquaredError)

# print (m1, m2, b)

# X = np.array([x1,x2]).T
# Y = np.array(y)

# model = LinearRegression()
# model.fit(X,Y)
# print(f"Optimal slope(m): {model.coef_[0]}")
# print(f"Optimal slope(m): {model.coef_[1]}")
# print(f"Optimal y-intercept(b): {model.intercept_}")

# print(abs(m1-model.coef_[0]) < 0.001)
# print(abs(m2-model.coef_[1]) < 0.001)
# print(abs(b-model.intercept_) < 0.001)

x1 = [1,2,3,4,5]
x2 = [3,5,6,8,11]
x3 = [7,2,9,4,6]
y = [1,4,6,7,10]
w = [0,0,0]
b = 0
learning_rate = .005
step = 0
X = np.array([x1, x2, x3]).T
n = X.shape[0]
k = X.shape[1]
errors = X @ w + b - y

def MSE(length, error):
    return ((1/length * error * error).sum())

grad_w = 2/n * X.T @ errors
grad_b = 2/n * errors.sum()

w = w - learning_rate*grad_w
b = b - learning_rate*grad_b

while (np.abs(grad_w).sum() + abs(grad_b) > 0.00001) and step <= 50000:
    errors = X @ w + b - y
    grad_w = 2/n * X.T @ errors
    grad_b = 2/n * errors.sum()

    w = w - learning_rate*grad_w
    b = b - learning_rate*grad_b
    step += 1
    meanSquaredError = MSE (n, errors)
    if (step%1000 == 1):
        print("step ", step, ": ", meanSquaredError)

print(w, b)