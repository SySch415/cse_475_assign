import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
plt.show()

diabetes_X, diabetes_y, = datasets.load_diabetes(return_X_y=True)
print('The shape of the input features:', diabetes_X.shape)
print('The shape of the output variables:', diabetes_y.shape)

diabetes_X_one = diabetes_X[:, np.newaxis, 2]
print(diabetes_X_one.shape)

diabetes_X_train = diabetes_X_one[:-20]
diabetes_X_test = diabetes_X_one[-20:]
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

print("Training input variable shape: ", diabetes_X_train.shape)
print("Test input variable shape: ", diabetes_X_test.shape)

print()

# Lease Squares section

bias = np.ones((422, 1))
design_X = np.hstack((bias, diabetes_X_train))
# print("Design:", design_X.shape)

diabetes_y_train_col = diabetes_y_train.reshape(-1, 1)
# print("y_train_col:", diabetes_y_train_col.shape)

design_dot_inverse = (np.linalg.inv(design_X.T @ design_X))
theta = design_dot_inverse @ design_X.T @ diabetes_y_train_col


# Gradient Descent Section

rand_theta = np.zeros((2, 1))
learn_rate = 0.03

for t in range(100000):
    loss = np.sum((design_X @ rand_theta - diabetes_y_train_col) ** 2)
    loss *= 1 / design_X.shape[0]
    grad_theta_L = (design_X @ rand_theta - diabetes_y_train_col)
    grad_theta_L = 2 / design_X.shape[0] * (design_X.T @ grad_theta_L)
    rand_theta = rand_theta - learn_rate * grad_theta_L

model = LinearRegression()
model.fit(diabetes_X_train, diabetes_y_train)
# print("Least Squares bias and theta 1:", theta)
print("Gradient Descent bias and theta 1:", rand_theta)
print("scikit bias:", model.intercept_)
print("scikit theta_1:", model.coef_)
