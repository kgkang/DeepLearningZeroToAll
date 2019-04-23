# udemy u31
# https://www.udemy.com/complete-guide-to-tensorflow-for-deep-learning-with-python/learn/lecture/7876634#overview

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pylab as plt


x_data = np.linspace(0,10.0, 1000000)
noise = np.random.randn(len(x_data))
y_data = (0.5 * x_data) + 5 + noise

x_df = pd.DataFrame(data=x_data, columns=['X data'])
y_df = pd.DataFrame(data=y_data, columns=['Y'])
# print(x_df.head)

my_data = pd.concat([x_df,y_df],axis=1)
# print(my_data.head)
# print(my_data.sample(n=250))

# my_data.sample(n=250).plot(kind='scatter', x="X data", y="Y")
# plt.show()

batch_size = 8

m = tf.Variable(0.81)
b = tf.Variable(0.17)

# train 루프마다 train data를 갱신시키기 위해 placeholder를 사용
xph = tf.placeholder(dtype=tf.float32, shape=[batch_size])
yph = tf.placeholder(dtype=tf.float32, shape=[batch_size])

# Model & Loss
# error = 0
# for x,y in zip(xph, yph):
#     y_model = m*x + b
#     error += (y-y_model)**2

y_model = m * xph + b
error = tf.reduce_sum(tf.square(yph-y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    batches = 1000
    for i in range(batches):
        batch_mask = np.random.randint(len(x_data),size=batch_size)
        feed = {xph: x_data[batch_mask], yph: y_data[batch_mask]}
        sess.run(train, feed_dict=feed)

    model_m, model_b = sess.run([m,b])

y_hat = x_data * model_m + model_b

my_data.sample(n=250).plot(kind='scatter', x="X data", y="Y")
plt.plot(x_data, y_hat, 'r')
plt.show()






