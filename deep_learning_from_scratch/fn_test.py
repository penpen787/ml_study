# 책에 나오는 습작용 공식들 정리
import numpy as np
import matplotlib.pylab as plt
import fn

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

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
# plt.show()

# x=5, x=10 일때 미분값 계산
print(numerical_diff_middle(function_1, 5))
print(numerical_diff_middle(function_1, 10))

'''
편미분 : partial derivative
'''
# f(x1, x2) = x1^2 + x2^2
def function_two_variable(x):
    return x[0]**2 + x[1]**2  # or np.sum(x**2)

# 상미분을 풀기 위해선 풀고자하는 하나의 변수만 남겨 논 후, 나머지는 상수처리함
# x1 = 3, x2 = 4 일때 x1 에 대해 편미분 하라
def function_x1(x1):
    return x1*x1 + 4.0**2.0

def function_x2(x2):
    return 3.0**2.0 + x2*x2

print(numerical_diff_middle(function_x1, 3.0))  # 6.00000000000378
print(numerical_diff_middle(function_x2, 4.0))  # 7.999999999999119

# 세점 (3,4), (0,2), (3,0) 에서 기울기
print(fn.numerical_gradient(function_two_variable, np.array([3.0, 4.0])))
print(fn.numerical_gradient(function_two_variable, np.array([0.0, 2.0])))
print(fn.numerical_gradient(function_two_variable, np.array([3.0, 0.0])))

# 경사하강법으로 f(x1, x2) = x1**2 + x2**2 의 최솟값을 구하라
def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])
g_d_test = fn.gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
print(g_d_test)

# 경사하강법 학습률이 너무 클때 & 너무 작을때
# 학습률이 너무 작거나 크면 너무 큰값으로 발산해버리거나 거의 갱신되지 않음
g_d_test2 = fn.gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100)
print(g_d_test2)
g_d_test3 = fn.gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100)
print(g_d_test3)

