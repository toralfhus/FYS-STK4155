import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from time import time

noise_coef = 3
x = np.random.rand(100, 1)
y = 2.0 + 5 * x * x + noise_coef * np.random.randn(100, 1)
print(x.shape, y.shape)


# 2.1 computing the parametrization of the data fitting a second-order polynomial
def betas_analytical(x, y):
    x_inv = np.linalg.inv(x.T.dot(x))
    betas = x_inv.dot(x.T).dot(y)
    # print(betas)
    return betas


# Creating the design matrix
X = np.empty(shape=(100, 3))
X[:, 0] = np.ones(100)
X[:, 1] = x[:, 0]
X[:, 2] = x[:, 0]**2

print(X)
betas = betas_analytical(X, y)
print(X.shape, betas.shape)
yhat = np.dot(X, betas)
print(yhat.shape)


lr = LinearRegression(fit_intercept=True)
lr.fit(X, y)
yhat_skl = lr.predict(X)


r2 = r2_score(y, yhat)
r2_skl = r2_score(y, yhat_skl)
mse = mean_squared_error(y, yhat)
mse_skl = mean_squared_error(y, yhat_skl)
print(f"r2={r2:.3g}, r2_skl={r2_skl:.3g}")
print(f"mse={mse:.3g}, mse_skl={mse_skl:.3g}")


from matplotlib import pyplot as plt
plt.plot(x, y, "x", label="Generated")
plt.plot(x, yhat, "o", label="Analytical", alpha=0.5)
plt.plot(x, yhat_skl, "o", label="Sklearn", alpha=0.5)
# plt.show()
plt.close()


print("\n\n--- Exercise 3 ---")
np.random.seed(2018)
n = 30
p_max = 15   # max degree of polynomial fit
test_frac = 0.20


# Make data set.
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)
print(x.shape, y.shape)

# setting up design matrix up to fifth-order polynomial
X = np.empty(shape=(n, p_max + 1))
X[:, 0] = np.ones(n)
for i in range(1, p_max + 1):
    X[:, i] = x[:, 0]**i
print(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac)
pvals = list(range(1, p_max + 2))
mse_vals = np.empty(shape=(p_max + 1, 2))


fig, ax = plt.subplots()
ax.plot(x, y, label="ground truth", zorder=100)


for p in pvals:
    print(f"p={p}:", end="\t")
    X_train_p = X_train[:, :p]
    X_test_p = X_test[:, :p]
    print(X_train_p.shape, X_test_p.shape, end="\t\t")


    time_0 = time()
    betas = betas_analytical(X_train_p, y_train)
    yhat_train = np.dot(X_train_p, betas)
    yhat_test = np.dot(X_test_p, betas)
    time_1 = time()


    print(f"Matrix mult: time={time_1 - time_0:.3e} s", end="\t")
    mse_train = mean_squared_error(y_train, yhat_train)
    mse_test = mean_squared_error(y_test, yhat_test)


    time_0 = time()
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X_train_p, y_train)
    yhat_skl_train = lr.predict(X_train_p)
    yhat_skl_test = lr.predict(X_test_p)
    time_1 = time()
    print(f"Sklearn: time={time_1 - time_0:.3e} s", end="\t")
    mse_skl_train = mean_squared_error(y_train, yhat_skl_train)
    mse_skl_test = mean_squared_error(y_test, yhat_skl_test)

    print(f"MSE train = {mse_train:.3f} / {mse_skl_train:.3f}\ttest = {mse_test:.3f} / {mse_skl_test:.3f}")
    mse_vals[p-1, 0] = mse_train
    mse_vals[p-1, 1] = mse_test


    # X_p = np.concatenate([X_train_p, X_test_p])
    # yhat_p = np.dot(X_p, betas)
    # ax.plot(x, yhat_p, ":", label=f"p={p}", alpha=0.5)
    yhat_p = np.dot(X_train_p, betas)
    ax.plot(X_train[:, 1], yhat_p, "x", label=f"p={p-1}", alpha=0.5)


fig.legend()


fig, ax = plt.subplots()
plt.plot(pvals, mse_vals[:, 0], "o", label="train")
plt.plot(pvals, mse_vals[:, 1], "x", label="test")
plt.xlabel("Polynomial degree $p$")
plt.ylabel("MSE")
plt.legend()
plt.grid()
plt.show()


# From https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/chapter1.html
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(2018)
n = 30
maxdegree = 14
# Make data set.
x = np.linspace(-3, 3, n).reshape(-1, 1)
y = np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)+ np.random.normal(0, 0.1, x.shape)
TestError = np.zeros(maxdegree)
TrainError = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


for degree in range(maxdegree):
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression(fit_intercept=False))
    clf = model.fit(x_train,y_train)
    y_fit = clf.predict(x_train)
    y_pred = clf.predict(x_test)
    polydegree[degree] = degree
    TestError[degree] = np.mean( np.mean((y_test - y_pred)**2) )
    TrainError[degree] = np.mean( np.mean((y_train - y_fit)**2) )

plt.plot(polydegree, TestError, label='Test Error')
plt.plot(polydegree, TrainError, label='Train Error')
plt.legend()
plt.show()