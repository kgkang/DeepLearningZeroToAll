import tensorflow as tf



with tf.Graph().as_default():
    # Create a six-element vector (1-D tensor).
    primes = tf.constant([2,3,5,7,11,13],dtype=tf.int32)

    # Create another six-element vector. Each element in the vector will be
    # initialized to 1. The first argument is the shape of the tensor (more
    # on shapes below).
    ones = tf.ones([6],dtype=tf.int32)

    # tf math API
    # https: // www.tensorflow.org / api_docs / python / tf / math
    just_add = tf.add(primes, ones)

    # Create a session to run the default graph.
    with tf.Session() as sess:
        print(just_add.eval())