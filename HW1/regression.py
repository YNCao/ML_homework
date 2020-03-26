import numpy as np
import matplotlib.pyplot as plt

# Step 1: load data
data = np.loadtxt('./data.txt')
x = data[:, 0]
y = data[:, 1]


# Step 2: choose and build model (coefficient matrix)
# model=1: linear model
# model=2: quadratic model
model = 1
if model == 1:
    A = np.ones((data.shape[0], 2))
    A[:, 1] = x
elif model == 2:
    A = np.ones((data.shape[0], 3))
    A[:, 1] = x
    A[:, 2] = x ** 2


# Step 3: calculate coefficients w = (A^T A)^-1 A^T y
c1 = np.linalg.inv(np.dot(A.transpose(), A))
c2 = np.dot(A.transpose(), y)
w = np.dot(c1, c2)


# Step 4: statistical analysis
y_ = np.dot(A, w)
# sum of squares for regression
SSR = np.sum((y_ - np.average(y))** 2)
# sum of squares for error
SST = np.sum((y - np.average(y))** 2)
# goodness of fit
R2 = SSR/SST


# Step 5: plot
# regression function
x_min = np.floor(np.min(x))
x_max = np.ceil(np.max(x))
xx = np.linspace(x_min, x_max, 100)
if model == 1:
    yy = w[0] + w[1] * xx
elif model == 2:
    yy = w[0] + w[1] * xx + w[2] * (xx ** 2)
# plot figure
plt.figure()
# plt.axes([x_min,x_max,0.2,1.6])
plt.ylim(0.3, 1.5)
plt.scatter(x, y, c='r', marker='o')
plt.plot(xx, yy)
plt.xlabel('x')
plt.ylabel('y')
if model == 1:
    plt.title('Linear Regression')
    plt.text(0.3, 0.4, '$y=w_0+w_1x$\n$w_0=$%.4f\n$w_1=$%.4f\n\n$R^2=$%.4f' % (w[0], w[1], R2), bbox=dict(alpha=0.2), fontsize=11)
    plt.savefig('linear.pdf')
elif model == 2:
    plt.title('Quadratic Regression')
    plt.text(0.3, 0.4, '$y=w_0+w_1x+w_2x^2$\n$w_0=$%.4f\n$w_1=$%.4f\n$w_2=$%.4f\n\n$R^2=$%.4f' % (w[0], w[1], w[2], R2), bbox=dict(alpha=0.2), fontsize=11)
    plt.savefig('quadratic.pdf')

plt.show()