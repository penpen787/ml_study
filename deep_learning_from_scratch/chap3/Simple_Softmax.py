import numpy as np

a = np.array([0.3, 2.9, 4.0])

# 지수값
exp_a = np.exp(a)
print(exp_a)

# 지수 함수의 합
sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

b = np.array([0.3, 2.9, 4.0])
print(softmax(b))