

import tensorflow as tf


print(tf.__version__)


hello = tf.constant('Hello Tensorflow')

sess = tf.Session()

hello_out = sess.run(hello)

print(hello_out)

sess.close()