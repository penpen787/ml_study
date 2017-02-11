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
