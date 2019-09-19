from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# Session test
# https://www.tensorflow.org/guide/low_level_intro#session

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
c = tf.constant("Hello ~~")

total = a + b

sess = tf.Session()
print(sess.run(total))

print(sess.run({'ab': (a,b), 'total': total}))

print(sess.run(c))



vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))

sess.close()