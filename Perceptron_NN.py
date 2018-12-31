import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=500, n_features=2, centers=2, random_state=11)


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def predict(x, weights):
    z = np.dot(x, weights)
    return sigmoid(z)


def loss(x, y, weights):
    y_ = predict(x, weights)
    cost = np.mean(-y * np.log(y_) - (1 - y) * np.log(1 - y_))
    return cost


def update(x, y, weights, learning_rate):
    y_ = predict(x, weights)
    dw = np.dot(x.T, y_ - y)
    weights = weights - learning_rate * dw / float(x.shape[0])
    return weights


def train(x, y, learning_rate=0.5, max_Epochs=100):
    ones = np.ones((x.shape[0], 1))
    x = np.hstack((ones, x))
    weights = np.zeros(x.shape[1])
    for epoch in range(max_Epochs):
        weights = update(x, y, weights, learning_rate)
        if epoch % 10 == 0:
            l = loss(x, y, weights)
            print('Epoch %d Loss %.4f' % (epoch, l))
    return weights


weights = train(x, y, learning_rate=0.8, max_Epochs=1000)


def getPrediction(x_test, weights, labels=True):
    if x_test.shape[1] != weights.shape[0]:
        ones = np.ones((x_test.shape[0], 1))
        x_test = np.hstack((ones, x_test))
    probs = predict(x_test, weights)
    if not labels:
        return probs
    else:
        labels = np.zeros(probs.shape)
        labels[probs >= 0.5] = 1
        return labels


x1 = np.linspace(-8, 2, 10)
x2 = -(weights[0] + weights[1] * x1) / weights[2]

y_ = getPrediction(x, weights)
print(np.sum(y_ == y) / y.shape[0])

plt.scatter(x[:, 0], x[:, 1], c=y)
plt.plot(x1, x2, color='k')
plt.show()
