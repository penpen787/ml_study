from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])
print("x_image=")
print(x_image)

# Weight(가중치행렬) and bias(편향) tf variable 초기화
def weight_variable(shape):
    # truncted_normal 은 값이 표준편차*2 보다 큰범위는 잘라냄
    # 임의잡음random noise) 셋팅
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # 초기값 0.1의 bias(b) 생성
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 스트라이드:1, padding은 기존이미지와 동일하게 'SAME')
def conv2d(x, W):
    # conv2d 는 input(x) 와 필터(W) 에 대해 2-D 합성곱을 함
    # strides 는 [batch, height, width, channels] 을 나타냄, batch? channels-RGB, grayscale(0,1)
    # padding:
    # VALID : only ever drops the right-most columns (or bottom-most rows)
    # SAME :  tries to pad evenly left and right
    # http://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# maxpooling
def max_pool_2x2(x):
    # ksize(kernal size) : The size of the window for each dimension of the input tensor.
    # pooling list : https://www.tensorflow.org/versions/r0.11/api_docs/python/nn/pooling
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

'''
 -- 모델 시작 --
 첫번째 합성곱계층 & 풀링 계층
 윈도 크기 5x5
 SO, 구조가 [5,5,1,32] 인 가중치 행렬이 필요함 / 5*5 -> pixel size, 1 -> channel, 32 -> feature maps
'''
W_conv1 = weight_variable([5, 5, 1, 32])
# 32개 특징맵(가중치행렬) 에 대한 편향
b_conv1 = bias_variable([32])

# 입력 이미지에 합성곱 적용 후, 활성함수(relu) 적용 -> maxpool
y_1 = conv2d(x_image, W_conv1) + b_conv1  # 계층1
h_conv1 = tf.nn.relu(y_1)  # 계층2 28x28
h_pool1 = max_pool_2x2(h_conv1)  # 계층3 14x14

'''
 두번째 합성곱계층 & 풀링계층
 W 의 5,5,32,64 의 이유는
 32 : 이전 계층 W_conv1 의 출력값 크기
 64 :64개의 특징맵
'''
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

y_2 = conv2d(h_pool1, W_conv2) + b_conv2  # 계층4
h_conv2 = tf.nn.relu(y_2)  # 계층5 14x14
h_pool2 = max_pool_2x2(h_conv2)  # 계층6 7x7

# 소트프맥스 계층에 주입하기 위해 7x7 출력값을 완전 열결 계층(fully connected layer) 에 연결
# 뉴런은 1024개 사용
# ?? 이부분 좀 이해 안감, 어떻게 7x7x64 -> 1024 개 뉴런으로 연결방법?
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# 텐서를 벡터로 변환
# 이전장처럼 소프트맥스는 이밎를 직렬화 해서 벡터형태로 입력해야 하기 때문
# 소프트맥스 p99, 는 각 클래스에 대한 확률을 얻게 해줌
# shape -1 '추론' 임, 즉 자동 배정
print(h_pool2)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
print(h_pool2_flat)
# h_pool2_flat, W_fc1 의 행렬곱 후, 뉴런과 더한 후 relu 연산 적용
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
# 오버피팅을 막기 위한 drop out 설정
# 오버피팅 : 학습데이터(training data) 에 대한 학습이기때문에 실제데이터와 안맞을 수 있음
# https://hunkim.github.io/ml/lec7.pdf
# 여러 방법이 있지만, regulation 사용
# 드롭아웃 : 신경망에서 뉴런에 해당하는값을 부분적으로 무시한다 확률로 지정
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

''' 모델 완료 '''
''' 훈련 & 평가 '''

# 오차확인 - 책 107p 참조
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# 경사하강법 대신, 아담 옵티마이저를 사용
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 모델 평가 - 정확도
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()

sess.run(tf.initialize_all_variables())
for i in range(200):
    # 배치단위로 실행
    batch = mnist.train.next_batch(50)
    if i%10 == 0:
        # 배치 단위의 비용(loss) 평가
        loss = sess.run(cross_entropy, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        # 배치 단위의 정확도평가
        train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, batch loss = %g, training accuracy %g"%(i, loss, train_accuracy))

    sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 테스트 set 결과에 대한 정확도
print("test accuracy %g"% sess.run(accuracy, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
