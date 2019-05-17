
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# print(type(mnist))

# numpy.ndarrary 타입의 mnist images
# print(mnist.train.images)
# image
# print(type(mnist.train.images))
# print(mnist.train.num_examples)
# print(mnist.test.num_examples)
# print(mnist.validation.num_examples)
# print(mnist.train.images[1].shape)
# print(type(mnist.train.images[1]))
# print(mnist.train.images[1].max())

import matplotlib.pylab as plt
# plt.imshow(mnist.train.images[1].reshape(28,28))
# plt.imshow(mnist.train.images[1].reshape(28,28),cmap='gist_gray')
# plt.show()

# Create Model
x = tf.placeholder(tf.float32, shape=[None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Create Graph
y = tf.matmul(x, W) + b

# Loss and Optimizer
y_true = tf.placeholder(tf.float32,[None,10])

# Cross Entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)


# Create Session
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_x, y_true: batch_y})

    # Test the train Model
    matches = tf.equal(tf.argmax(y,1), tf.argmax(y_true,1))

    acc = tf.reduce_mean(tf.cast(matches, tf.float32))

    print(sess.run(acc, feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))

