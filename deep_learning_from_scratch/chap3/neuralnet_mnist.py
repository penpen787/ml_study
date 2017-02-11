import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from mnist import load_mnist
import fn

# data load
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test

# 학습된 가중치 매개변수를 읽음
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = fn.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = fn.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = fn.softmax(a3)

    return y

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)  # 확률이 가장 높은 원소의 인덱스를 얻음
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy: " + str(float(accuracy_cnt) / len(x)))

'''
batch 처리
여기서 말하는 batch 는 묶음 데이터를 의미하여 IO 연산이 CPU 연산에 비해 느리므로
다량의 이미지를 한번에 로딩하여 한번에 연산 처리함
'''
xb, tb = get_data()
network_b = init_network()

batch_size = 100
accuracy_cnt_b = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network_b, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt_b += np.sum(p == t[i:i+batch_size])

print("Accuracy: " + str(float(accuracy_cnt_b) / len(xb)))



