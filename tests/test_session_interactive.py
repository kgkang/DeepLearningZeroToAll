import tensorflow as tf

sess = tf.InteractiveSession()

r3_tensor = tf.ones([3,4,5])

# Reshape existing content into a 6x10 matrix
r3_tensor_A = tf.reshape(r3_tensor,[6,10])

# with sess.as_default():
#     assert tf.get_default_session() is sess
#     print(r3_tensor.eval())

print(r3_tensor.eval())
# print(r3_tensor_C.eval(session=sess))

