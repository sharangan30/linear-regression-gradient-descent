Linear Regression with Gradient Descent, From Scratch

Multivariable linear regression trained with batch gradient descent, written from scratch with NumPy. I wrote the loss, the gradients and the update rule by hand, then checked the result against scikit-learn's LinearRegression.

How it works

The model predicts X @ w + b, where X has one row per example and one column per input. At each step:

Compute the errors: errors = X @ w + b - y
Compute the mean squared error: MSE = (1/n) * sum(errors^2)
Compute the gradients: grad_w = (2/n) * X.T @ errors and grad_b = (2/n) * sum(errors)
Update: w = w - learning_rate * grad_w and b = b - learning_rate * grad_b

Training stops when the gradients are very small (their absolute values sum to less than tolerance) or when max_steps is reached.

Usage
python
w, b, losses = gradient_descent(X, y, learning_rate=0.02)

gradient_descent(X, y, learning_rate=0.005, max_steps=50000, tolerance=0.00001) works with any number of input columns. It returns the weights, the bias, and the MSE at every step, so the training can be plotted.

Test data

Six examples with three inputs, generated from known values: y = 2*x1 + 3*x2 - 1*x3 + 5. The correct answer is weights [2, 3, -1] and bias 5.

Results

I compared four learning rates on the same data. The plot shows the MSE over training steps on log scales.

Learning rate	Steps run	Result
0.005	50,000 (hit the cap)	Close, but not converged: weights about [2.012, 2.991, -0.993]
0.01	49,731	Converged
0.02	24,864	Converged
0.05	n/a	Diverged (values overflow to nan)
A larger learning rate converges in fewer steps, up to a limit. Past it, the updates overshoot and the loss grows instead of shrinking. On this data that limit is around 0.01 to 0.02, so 0.05 blows up.
The learning rate changes how fast training gets there, or whether it diverges, but not where the minimum is.
Verified against scikit-learn

With a learning rate of 0.02, my weights and bias match sklearn.linear_model.LinearRegression:

	Weights	Bias
scikit-learn	[2, 3, -1]	5.0
This project (0.02)	[2.0001, 2.9999, -0.9999]	4.9997

The largest weight difference is about 1e-4 and the bias differs by about 3e-4. The small gap exists because gradient descent stops at a tolerance, while scikit-learn solves for the weights directly.

Run it
pip install -r requirements.txt
python gradient_descent.py

Close the plot window to let the script finish and print the scikit-learn comparison.

What I learned
Why the gradient is X.T @ errors, and how the matrix shapes line up
How learning rate affects speed and stability
That a result should be checked against a trusted reference before it's believed