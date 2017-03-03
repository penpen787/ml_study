import sys, os
sys.path.append(os.pardir)
import numpy as np
import fn

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)  # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = fn.softmax(z)
        loss = fn.cross_entropy_error(y, t)

        return loss

net = SimpleNet()
print(net.W)  # 램덤 & 정규화된 가중치 매개변수

x =  np.array([0.6, 0.9])
p = net.predict(x)
print(p)

print(np.argmax(p))

t = np.array([0, 0, 1])
net.loss(x, t)

# f = lambda w: net.loss(x, t)
# dW = fn.numerical_gradient(f, net.W)