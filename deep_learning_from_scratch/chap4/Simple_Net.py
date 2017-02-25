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


