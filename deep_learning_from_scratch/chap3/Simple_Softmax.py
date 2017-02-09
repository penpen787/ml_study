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

# softmax 함수를 사용하는 이유는, 1. 모든 결과의 합이 1이되게함, 2. 지수함수를 사용하여 입력데이터의 각 원소의 차이보다 크게 함
c = np.array([4, 6])
print(softmax(c))
# 0.11:0.88, 원본은 4:6이었지만 지수함수를 사용하여 결과가 극명해짐

# 지수함수에서는 x가 조금만 커져도 overflow 가 날수 있어, input 에 최댓값을 빼준다.
# 참고로 소프트맥스에서 x에 어떤값을 더하거나 빼도 결과는 동일하다. 증명 : p93
'''  overflow 발생
d = np.array([1006, 998])
print(softmax(d))
'''

e = np.array([100, 50])
print(softmax(e))  # [  1.00000000e+00   1.92874985e-22]

eMax = np.max(e)
print(softmax(e - eMax))  # [  1.00000000e+00   1.92874985e-22]

# 개선된 softmax
def enhanced_softmax(a):
    max_value = np.max(a)
    exp_a = np.exp(a - max_value)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a
