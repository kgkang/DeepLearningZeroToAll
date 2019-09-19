import tensorflow as tf

r3_tensor = tf.ones([3,4,5])

# Reshape existing content into a 6x10 matrix
r3_tensor_A = tf.reshape(r3_tensor,[6,10])

#  Reshape existing content into a 3x20
# matrix. -1 tells reshape to calculate
# the size of this dimension.
r3_tensor_B = tf.reshape(r3_tensor, [3,-1])

# Reshape existing content into a
# 4x3x5 tensor
r3_tensor_C = tf.reshape(r3_tensor, [4,3,-1])

# Note that the number of elements of the reshaped Tensors has to match the
# original number of elements. Therefore, the following example generates an
# error because no possible value for the last dimension will match the number
# of elements.
# r3_tensor_error = tf.reshape(r3_tensor, [13, 2, -1])  # ERROR!

sess = tf.Session()
with sess.as_default():
    assert tf.get_default_session() is sess
    print(r3_tensor_C.eval())

# print(r3_tensor_B.eval())
# print(r3_tensor_C.eval(session=sess))

