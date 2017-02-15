# 책에 나오는 습작용 공식들 정리
import numpy as np

# 미분의 나쁜 예
# h 를 가급적 작은 값을 넣고 싶어 10e-50 을 대입함
# but 이 방식은 반올림 오차 문제를 일으킴
def numerical_dif(f, x):
    h = 10e-50
    return (f(x + h) - f(x)) / h

print(np.float32(10e-50))  # 0.0, 너무 작은값을 쓰면 컴퓨터로 계산하는데 문제가 됨
# h 를 10e-4 로 변경함
print(np.float32(10e-4))


# 중심차분 or 중앙차분 : x 를 중심으로 그 전후의 차분을 계산함 (x+h), (x-h)
def numerical_diff_middle(f, x):
    h = 1e-4  # 0.001
    return (f(x + h) - f(x - h)) / (2*h)

# 수치 미분의 예
# y = 0.01x^2 + 0.1x
def function_1(x):
    return 0.01*x**2 + 0.1*x

import matplotlib.pylab as plt

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

# x=5, x=10 일때 미분값 계산
print(numerical_diff_middle(function_1, 5))
print(numerical_diff_middle(function_1, 10))