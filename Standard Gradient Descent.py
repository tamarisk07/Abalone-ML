import KFold
import numpy as np
import Normalization as nml
# import matplotlib.pyplot as plt

#######################
X_train = KFold.X_train
Y_train = KFold.Y_train
X_test = KFold.X_test
Y_test = KFold.Y_test

X_mean_train = nml.X_mean_train
X_std_train = nml.X_Std_train
X_min_max_train = nml.X_min_max_train

#######################
# x, y could be X[i], Y[i]. They are both ndarrays.
# y:vector,(3759*1)   x:matrix,(3759*8)   theta:(8*1)     b:(3759*1)
# Calculate the cost function of data set(x,y) parameterized by theta and b.
def cost_function(x, y, theta, b):
    # Y_predicted:(3759*1)  m is the size of the data set
    m = y.size
    y = y.reshape(m, 1)
    y_predicted = np.dot(x, theta) + b
    y_gap = y_predicted - y
    # print(y_gap.shape)
    # print(y_predicted.shape)
    # print(y.shape)
    # print(x.shape)
    # print(theta.shape)
    return np.sum(np.square(y_gap)) / (2 * m)


#######################
# Calculate theta_gradient and b_gradient of the cost function with each theta and b
def gradient_function(theta, b, x, y):
    m = y.size
    y_gap = np.dot(x, theta) + b - y
    theta_grad = np.dot(x.T, y_gap) / m
    b_grad = np.dot(np.ones(m).reshape(1, m), y_gap) / m
    # print(x.shape)
    # print(y.shape)
    # print(y_gap.shape)
    # print(theta_grad.shape)
    # print(theta.shape)
    # print(b_grad.shape)
    return [theta_grad, b_grad]


#######################
# Given initialized 'theta' and 'b', use the gradient descent algorithm to find the optimal 'theta' and 'b'
# which makes the gradient to be small(smaller than the given epsillon) in all dimensions. Cost array is an
# array recording the cost of each iteration.
def gradient_descent(x, y, alpha, epsillon=0.1, theta=np.ones(8).reshape(8, 1), b=1, iteration_num=0):
    m = y.size
    y = y.reshape(m, 1)
    theta_grad, b_grad = gradient_function(theta, b, x, y)
    # initilize cost_array and gives the first element
    init_cost = cost_function(x, y, theta, b)
    cost_array = [init_cost]
    # choose to use GDA by iterations or epsillon
    if iteration_num > 0:
        for i in range(1, iteration_num+1):
            theta = theta - alpha * theta_grad
            b = b - alpha * b_grad
            theta_grad, b_grad = gradient_function(theta, b, x, y)
            # print('No. of iterations: %d' %i)
            # print(theta_grad)
            # print(b_grad)
            cur_cost = cost_function(x, y, theta, b)
            # print(cur_cost)
            cost_array.append(cur_cost)
    else:
        # i = 1
        while np.abs(theta_grad).sum() + abs(b_grad) > epsillon:
            theta = theta - alpha * theta_grad
            b = b - alpha * b_grad
            theta_grad, b_grad = gradient_function(theta, b, x, y)
            # print('No. of iterations: %d' % i)
            # print(theta_grad)
            # print(b_grad)
            cur_cost = cost_function(x, y, theta, b)
            # print(cur_cost)
            cost_array.append(cur_cost)
            # i += 1
    return [theta, b, cost_array]


#######################
# plt.figure(1)
def run(x_train, y_train, x_test, y_test):
    theta_optimal = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    b_optimal = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cost_iter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for j in range(10):
        theta_optimal[j], b_optimal[j], cost_iter[j] = gradient_descent(x_train[j], y_train[j], alpha=0.1, iteration_num=1000)
        # index = np.arange(len(cost_iter[j]))
        # plt.plot(index, cost_iter[j])
        # plt.show()
        # print('parameters_optimal: ')
        # print(theta_optimal)
        # print(b_optimal)
        # print(' ')
    print('######################################')
    print('Cost: ')
    for k in range(10):
        print('###############')
        print('for test group %d, we have the cost of train sets and test sets as below: ' % (k + 1))
        print(cost_iter[k][1000])
        print(cost_function(x_test[k], y_test[k], theta_optimal[k], b_optimal[k]))
        print()


run(X_train, Y_train, X_test, Y_test)
run(X_min_max_train, Y_train, X_test, Y_test)
run(X_mean_train, Y_train, X_test, Y_test)
run(X_std_train, Y_train, X_test, Y_test)
