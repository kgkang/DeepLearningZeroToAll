import tensorflow as tf


g = tf.Graph()

with g.as_default():
     x = tf.constant(8, name="x_const")
     y = tf.constant(5, name="y_const")
     z = tf.constant(4, name="z_const")
     sum1 = tf.add(x,y,name="sum1")
     sum2 = tf.add(sum1,z,name="sum2")

     with tf.Session() as sess:
         # sess.run(sum1)
         print(sum2.eval())