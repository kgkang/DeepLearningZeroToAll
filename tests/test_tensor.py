import tensorflow as tf


r3_tensor = tf.ones([3,4,5])

print(type(r3_tensor.shape))
# <class 'tensorflow.python.framework.tensor_shape.TensorShapeV1'>
print(r3_tensor.shape)
# (3, 4, 5)
print(r3_tensor.shape[0])
# 3
print(r3_tensor.shape[1])
# 4

zeros = tf.zeros(r3_tensor.shape[1])
print(zeros)
# Tensor("zeros:0", shape=(4,), dtype=float32)

## Changing the shape of a tf.Tensor
#
print(zeros.eval())


