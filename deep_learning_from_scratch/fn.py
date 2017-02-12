# common functions
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 개선된 softmax
def softmax(a):
    max_value = np.max(a)
    exp_a = np.exp(a - max_value)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a

# 교차 엔트로피 오차 (CEE)
def cross_entropy_error(y, t):
    delta = 1e-7  # np.log 에  0 입력시, 마이너스 무한대(-inf)가 출력되기 때문에, 아주 작은 값을 더해 0이 되지 않도록 함
    return -np.sum(t * np.log(y + delta))

