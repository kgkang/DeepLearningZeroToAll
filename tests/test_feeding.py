from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# https://www.tensorflow.org/guide/low_level_intro#feeding

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = tf.constant(3.0, dtype=tf.float32)
z = a + b + c

with tf.Session() as sess:
    print(sess.run(z, feed_dict={a:3, b:4}))
    print(sess.run(z, feed_dict={a:[1,3], b:[2,4]}))
    print(sess.run(z, feed_dict={a:3, b:4, c:1}))

