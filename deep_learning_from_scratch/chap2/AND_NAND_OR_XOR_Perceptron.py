import numpy as np

# W, weight : 가중치 - 각 입력 신호가 결과에 주는 영향력(중요도)을 조절하는 매개변수
# 참고, 편향 : 한쪽으로 치우쳐 균형을 깸
# b, bias : 편향 - 뉴런이 얼마나 쉽게 활성화(결과로 1을 출력) 하느냐를 조정하는 매개변수
def AND_THEORY(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7 # Weights, threshold
    val = x1*w1 + x2*w2
    if val <= theta:
        return 0
    else:
        return 1

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7 # bias = (-threshold)
    val = np.sum(w*x) + b
    if val <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    val = np.sum(w*x) + b
    if val <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.3
    val = np.sum(w*x) + b
    if val <= 0:
        return 0
    else:
        return 1

# XOR 은 1-layer : NAND, OR, 2-layer : AND 로 표현 가능
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)

print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))

