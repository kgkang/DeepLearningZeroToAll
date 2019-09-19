from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # tf.float32 가 묵시적으로 적용

total = a + b

print(a)
print(b)
print(total)
#
# # ==>  3,4,7을 출력하지 않음. 각각의 출력 텐서를 표시
# # Tensor("Const:0", shape=(), dtype=float32)
# # Tensor("Const_1:0", shape=(), dtype=float32)
# # Tensor("add:0", shape=(), dtype=float32)

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
writer.flush()