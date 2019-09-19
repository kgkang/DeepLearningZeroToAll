from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# https://www.tensorflow.org/guide/low_level_intro#datasets

my_data = [2,3,4,5,6]

slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

sess = tf.Session()

while True:
    try:
        print(sess.run(next_item))
    except tf.errors.OutOfRangeError:
        break

r = tf.random_normal([10,3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

sess.run(iterator.initializer)
while True:
    try:
        print(sess.run(next_row))
    except tf.errors.OutOfRangeError:
        break

# print(sess.run(r))

sess.close()

# writer = tf.summary.FileWriter(".")
# writer.add_graph(tf.get_default_graph())
# writer.flush()