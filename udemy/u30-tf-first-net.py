import tensorflow as tf
import numpy as np

n_features = 10
n_neurons = 3


x = tf.placeholder(tf.float32,[None,n_features])
W = tf.Variable(tf.random_normal([n_features,n_neurons]))
b = tf.Variable(tf.ones([n_neurons]))

xW = tf.matmul(x, W)
z = tf.add(xW, b)

a = tf.sigmoid(z)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    out = sess.run(a,feed_dict={x:np.random.random([3, n_features])})

print(out)

