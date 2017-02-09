import sys, os
sys.path.append(os.pardir)
from mnist import load_mnist

# mnist 다운로드
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# 각 데이터의 형상 출력
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)