import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from mnist import load_mnist
import fn

(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True, normalize=True)

print(x_train.shape)
print(t_train.shape)

# 무작위 10개만 추출
# ex
print(np.random.choice(60000, 10))

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

