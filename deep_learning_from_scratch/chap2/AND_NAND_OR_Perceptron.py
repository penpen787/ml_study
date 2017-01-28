import numpy as np

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
