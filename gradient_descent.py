import numpy as np
from sklearn.linear_model import LinearRegression

x = [1,2,3,4,5]
y = [1,4,6,7,10]
m = 0
b = 0
n = len(x)
learning_rate = .01
step = 0

# Functions
def error(x, y, m, b):
    return (m*x+b-y)
def MSE(length, errorList):
    value = 0
    for i in range(length):
        value += errorList[i]*errorList[i]
    return (1/n*value)
def dervB(errorList, length):
    return (2/length*(sum(errorList)))
def dervM(errorList, length, x):
    value = 0
    for i in range(length):
        value += x[i]*errorList[i]
    return (2/n*value)

errorList = []
for i in range(n):
    errorList.append(error(x[i], y[i], m, b))
print(errorList)

dervWRT_B = dervB(errorList, n)
print(dervWRT_B)

dervWRT_M = dervM(errorList, n, x)
print(dervWRT_M)

m = m - learning_rate * dervWRT_M
b = b - learning_rate * dervWRT_B
step += 1
meanSquaredError = MSE(n, errorList)
print("step ", step, ": ", meanSquaredError)

while (((abs(dervWRT_B) + abs(dervWRT_M)) > 0.00001) and step <= 5000):

    errorList = []
    for i in range(n):
        errorList.append(error(x[i], y[i], m, b))
    dervWRT_B = dervB(errorList, n)
    dervWRT_M = dervM(errorList, n, x)
    m = m - learning_rate * dervWRT_M
    b = b - learning_rate * dervWRT_B
    step += 1
    meanSquaredError = MSE (n, errorList)
    print("step ", step, ": ", meanSquaredError)

print (m, b)

X = np.array(x).reshape(-1,1)
Y = np.array(y)

model = LinearRegression()
model.fit(X, Y)
print(f"Optimal slope(m): {model.coef_[0]}")
print(f"Optimal y-intercept(b): {model.intercept_}")

print(abs(m-model.coef_[0]) < 0.001)
print(abs(b-model.intercept_) < 0.001)