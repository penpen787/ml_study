import initPoints as ip
import tensorflow as tf

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * ip.x_data + b

print(y)

loss = tf.reduce_mean(tf.square(y - ip.y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(16):
    sess.run(train)

print(sess.run(W), sess.run(b))

ip.plt.plot(ip.x_data, sess.run(W) * ip.x_data + sess.run(b))
ip.plt.legend()
ip.plt.xlabel('x')
ip.plt.ylabel('y')
ip.plt.show()
